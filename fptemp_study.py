import numpy as np
from numpy.random import RandomState
from Quaternion import Quat
from astropy.io import ascii
import xija
from astropy.constants import R_earth
from astropy.coordinates import solar_system_ephemeris, \
    get_body, SkyCoord
from cxotime import CxoTime
import Ska.Sun
import Ska.quatutil
import Ska.astro
import h5py

solar_system_ephemeris.set('jpl')

R_e = R_earth.to_value("km")

msid = {"dpa": "1dpamzt",
        "acisfp": "fptemp"}


def make_states(start, stop, ccd_count, q, fep_count=None, clocking=1,
                simpos=75624.0):
    if fep_count is None:
        fep_count = ccd_count
    tstart = float(CxoTime(start).secs)
    tstop = float(CxoTime(stop).secs)
    states = {"ccd_count": np.array([ccd_count], dtype='int'),
              "fep_count": np.array([fep_count], dtype='int'),
              "clocking": np.array([clocking], dtype='int'),
              "vid_board": np.array([ccd_count > 0], dtype='int'),
              "simpos": np.array([simpos]),
              "tstart": np.array([tstart]),
              "tstop": np.array([tstop]),
              "q1": np.array([q[0]]),
              "q2": np.array([q[1]]),
              "q3": np.array([q[2]]),
              "q4": np.array([q[3]]),
              }
    dtype = [(k, str(v.dtype)) for k, v in states.items()]
    data = np.zeros(1, dtype=dtype)
    for k, v in states.items():
        data[k] = v
    return data


def calc_pitch_roll(times, sun_eci, chandra_eci, states):
    """Calculate the normalized sun vector in body coordinates.
    Shamelessly copied from Ska.engarchive.derived.pcad but 
    modified to use commanded states quaternions

    Parameters
    ----------
    times : NumPy array of times in seconds
    ephem : orbitephem and solarephem info 
    states : commanded states NumPy recarray

    Returns
    -------
    3 NumPy arrays: time, pitch and roll
    """
    from Ska.engarchive.derived.pcad import arccos_clip, qrotate
    idxs = Ska.Numpy.interpolate(np.arange(len(states)), states['tstart'],
                                 times, method='nearest')
    states = states[idxs]

    sun_vec = -chandra_eci + sun_eci
    est_quat = np.array([states['q1'],
                         states['q2'],
                         states['q3'],
                         states['q4']])

    sun_vec_b = qrotate(est_quat, sun_vec)  # Rotate into body frame
    magnitude = np.sqrt((sun_vec_b ** 2).sum(axis=0))
    magnitude[magnitude == 0.0] = 1.0
    sun_vec_b = sun_vec_b / magnitude  # Normalize

    pitch = np.degrees(arccos_clip(sun_vec_b[0, :]))
    roll = np.degrees(np.arctan2(-sun_vec_b[1, :], -sun_vec_b[2, :]))

    return pitch, roll


def generate_targets(tstart, num_targs, prng=None):
    if prng is None:
        prng = np.random
    elif isinstance(prng, int):
        prng = RandomState(prng)

    ra_t = np.array([])
    dec_t = np.array([])

    t = CxoTime(tstart)
    sun = get_body('sun', t)

    while len(ra_t) < num_targs:
        ra = 360.0*prng.uniform(low=0.0, high=1.0, size=num_targs)
        dec = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_targs))
        dec = np.rad2deg(dec)-90.0
        tcoord = SkyCoord(ra=ra, dec=dec, unit='deg')
        pitch = sun.separation(tcoord).to_value("deg")
        good = (pitch > 46.2) & (pitch < 177.9)
        ra_t = np.append(ra_t, tcoord.ra.to_value("deg")[good])
        dec_t = np.append(dec_t, tcoord.dec.to_value("deg")[good])

    sun_ra = sun.ra.to_value('deg')
    sun_dec = sun.dec.to_value('deg')

    roll_t = [
        Ska.Sun.nominal_roll(ra, dec, sun_ra=sun_ra, sun_dec=sun_dec)
        for ra, dec in zip(ra_t, dec_t)
    ]

    eq = np.array([ra_t, dec_t, roll_t]).T
    q = Quat(equatorial=eq)

    return q


class RunFPTempModels:
    def __init__(self, tstart=None, tstop=None, model_spec=None):
        if model_spec is None:
            model_spec = 'acisfp_model_spec.json'
        self.ephem_table = ascii.read("ephem2023.dat")
        self.ephem_times = CxoTime(self.ephem_table["dates"].data).secs
        if tstart is None:
            tstart = self.ephem_times[0]
        if tstop is None:
            tstop = self.ephem_times[-1]
        self.model_spec = model_spec
        self._get_orbits(tstart, tstop)
        self.num_orbits = self.per_times.size
        self.tstart = tstart
        self.tstop = tstop

    def _get_ephem(self, tstart, tstop):
        tstart = float(CxoTime(tstart).secs)
        tstop = float(CxoTime(tstop).secs)
        idxs = np.searchsorted(self.ephem_times, [tstart-2000.0, tstop+2000.])
        solarephem = np.array(
            [self.ephem_table[f"solarephem0_{ax}"].data[idxs[0]:idxs[1]] for ax in "xyz"]
        )
        orbitephem = np.array(
            [self.ephem_table[f"orbitephem0_{ax}"].data[idxs[0]:idxs[1]] for ax in "xyz"]
        )
        return self.ephem_times[idxs[0]:idxs[1]], solarephem*1000.0, orbitephem*1000.0

    def _get_orbits(self, tstart, tstop):
        from scipy.signal import find_peaks
        times, _, orbitephem = self._get_ephem(tstart, tstop)
        r = np.sqrt((orbitephem**2).sum(axis=0))
        idxs, _ = find_peaks(1.0/r)
        self.per_times = times[idxs]
        self.per_dates = CxoTime(self.per_times).date
        self.r_perigee = r[idxs]*1.0e-3
        t = ascii.read("rad_zones.orp")
        self.el_dates = np.array([[t["GMT"][i], t["GMT"][i+1]]
                                  for i in range(0, len(t), 2)])
        self.el_times = CxoTime(self.el_dates).secs

    @property
    def a_perigee(self):
        return self.r_perigee - R_e

    def calc_model(self, name, tstart, tstop, states, T_init=-119.8):
        model = xija.ThermalModel(name, start=tstart, stop=tstop,
                                  model_spec=self.model_spec)
        ephem_times, solarephem, orbitephem = self._get_ephem(tstart, tstop)
        state_times = np.array([states['tstart'], states['tstop']])
        model.comp['sim_z'].set_data(states['simpos'], state_times)
        model.comp['eclipse'].set_data(False)
        for name in ('ccd_count', 'fep_count', 'vid_board', 'clocking'):
            model.comp[name].set_data(states[name], state_times)
        pitch, roll = calc_pitch_roll(ephem_times, solarephem, orbitephem, states)
        model.comp['roll'].set_data(roll, ephem_times)
        model.comp['pitch'].set_data(pitch, ephem_times)
        model.comp[msid[name]].set_data(T_init, None)
        model.comp['dpa_power'].set_data(0.0)

        if name == "acisfp":
            model.comp['dh_heater'].set_data(0.0, model.times)
            for i in range(1, 5):
                model.comp[f'aoattqt{i}'].set_data(states[f'q{i}'], state_times)

            for i, axis in enumerate("xyz"):
                model.comp[f'orbitephem0_{axis}'].set_data(orbitephem[i,:], ephem_times)

            model.comp['1cbat'].set_data(-53.0)
            model.comp['sim_px'].set_data(T_init)
        elif name == "dpa":
            model.comp["dpa0"].set_data(T_init)

        model.make()
        model.calc()

        return model

    def _run(self):
        pass

    def run_perigees(self, ntargs=100, T_init=-119.8):
        pad_time = 6000.0
        for i in range(0, self.num_orbits):
            tstart = self.el_times[i, 0]-pad_time
            tstop = self.el_times[i, 1]+pad_time
            datestart = CxoTime(tstart).date
            datestop = CxoTime(tstop).date
            b = datestart[:8].replace(":", "_")
            e = datestop[:8].replace(":", "_")
            T = f"m{np.abs(np.round(T_init).astype('int'))}"
            f = h5py.File(f"acisfp_model_perigee_{b}_{e}_{T}.h5", "w")
            f.attrs['tstart'] = tstart
            f.attrs['tstop'] = tstop
            f.attrs['datestart'] = datestart
            f.attrs['datestop'] = datestop
            q = generate_targets(tstart, ntargs)
            f.create_dataset("q", data=q.q)
            t = []
            mvals = []
            ephem_t = []
            ephem_x = []
            ephem_y = []
            ephem_z = []
            pitch = []
            roll = []
            esa = []
            for k in range(q.shape[0]):
                states = make_states(tstart, tstop, 0, q.q[k, :],
                                     fep_count=3, clocking=0, 
                                     simpos=-99616.0)
                model = self.calc_model("acisfp", tstart, tstop, states, 
                                        T_init=T_init)
                t.append(model.times)
                mvals.append(model.comp["fptemp"].mvals)
                ephem_t.append(model.comp["orbitephem0_x"].times)
                ephem_x.append(model.comp["orbitephem0_x"].dvals)
                ephem_y.append(model.comp["orbitephem0_y"].dvals)
                ephem_z.append(model.comp["orbitephem0_z"].dvals)
                pitch.append(model.comp["pitch"].dvals)
                roll.append(model.comp["roll"].dvals)
                esa.append(model.comp['earthheat__fptemp'].dvals)
            f.create_dataset("times", data=np.array(t))
            f.create_dataset("fptemp", data=np.array(mvals))
            f.create_dataset("ephem_t", data=np.array(ephem_t))
            f.create_dataset("ephem_x", data=np.array(ephem_x))
            f.create_dataset("ephem_y", data=np.array(ephem_y))
            f.create_dataset("ephem_z", data=np.array(ephem_z))
            f.create_dataset("pitch", data=np.array(pitch))
            f.create_dataset("roll", data=np.array(roll))
            f.create_dataset("earth_solid_angle", data=np.array(esa))
            f.flush()
            f.close()

    def run_observations(self, exp_time=30.0, ntargs=100, T_init=-119.8):
        exp_time *= 1000.0
        for i in range(0, self.num_orbits-1):
            times = np.arange(self.per_times[i], self.per_times[i+1], exp_time)
            datestart = CxoTime(self.per_times[i]).date
            datestop = CxoTime(self.per_times[i+1]).date
            b = datestart[:8].replace(":", "_")
            e = datestop[:8].replace(":", "_")
            T = f"m{np.abs(np.round(T_init).astype('int'))}"
            f = h5py.File(f"acisfp_model_orbit_{b}_{e}_{T}.h5", "w")
            f.attrs['tstart'] = self.per_times[i]
            f.attrs['tstop'] = self.per_times[i+1]
            nobs = len(times)
            f.attrs['datestart'] = datestart
            f.attrs['datestop'] = datestop
            print(f"Running orbit from {datestart} to {datestop}.")
            for j in range(nobs):
                g = f.require_group(f"obs_{j}")
                g.attrs['tstart'] = times[j]
                g.attrs['tstop'] = times[j]+exp_time
                g.attrs['datestart'] = CxoTime(times[j]).date
                g.attrs['datestop'] = CxoTime(times[j]+exp_time).date
                q = generate_targets(times[j], ntargs)
                t = []
                mvals = []
                ephem_t = []
                ephem_x = []
                ephem_y = []
                ephem_z = []
                pitch = []
                roll = []
                esa = []
                g.create_dataset("q", data=q.q)
                for k in range(q.shape[0]):
                    states = make_states(times[j], times[j]+exp_time, 4, q.q[k,:])
                    model = self.calc_model("acisfp", times[j], times[j]+exp_time,
                                            states, T_init=T_init)
                    t.append(model.times)
                    mvals.append(model.comp["fptemp"].mvals)
                    ephem_t.append(model.comp["orbitephem0_x"].times)
                    ephem_x.append(model.comp["orbitephem0_x"].dvals)
                    ephem_y.append(model.comp["orbitephem0_y"].dvals)
                    ephem_z.append(model.comp["orbitephem0_z"].dvals)
                    pitch.append(model.comp["pitch"].dvals)
                    roll.append(model.comp["roll"].dvals)
                    esa.append(model.comp['earthheat__fptemp'].dvals)
                g.create_dataset("times", data=np.array(t))
                g.create_dataset("fptemp", data=np.array(mvals))
                g.create_dataset("ephem_t", data=np.array(ephem_t))
                g.create_dataset("ephem_x", data=np.array(ephem_x))
                g.create_dataset("ephem_y", data=np.array(ephem_y))
                g.create_dataset("ephem_z", data=np.array(ephem_z))
                g.create_dataset("pitch", data=np.array(pitch))
                g.create_dataset("roll", data=np.array(roll))
                g.create_dataset("earth_solid_angle", data=np.array(esa))

            f.flush()
            f.close()


if __name__ == "__main__":
    runner = RunFPTempModels("2021:001:00:00:00", "2023:365:23:59:59",
                             model_spec="acisfp_model_spec.json")
    #runner.run_observations()
    runner.run_perigees(T_init=-109.0)