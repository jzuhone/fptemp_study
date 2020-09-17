import numpy as np
from numpy.random import RandomState
from Quaternion import Quat
from astropy.io import ascii
import xija
from astropy.coordinates import solar_system_ephemeris, \
    get_body, SkyCoord
from cxotime import CxoTime
import Ska.Sun
import Ska.quatutil
from chandra_models import get_xija_model_file

solar_system_ephemeris.set('jpl')


def make_states(start, stop, ccd_count, q):
    tstart = float(CxoTime(start).secs)
    tstop = float(CxoTime(stop).secs)
    states = {"ccd_count": np.array([ccd_count], dtype='int'),
              "fep_count": np.array([ccd_count], dtype='int'),
              "clocking": np.array([1], dtype='int'),
              "vid_board": np.array([1], dtype='int'),
              "simpos": np.array([75624.0]),
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

    targets = np.array([])

    t = CxoTime(tstart)
    sun = get_body('sun', t)

    while len(targets) < num_targs:
        ra = 360.0*prng.uniform(low=0.0, high=1.0, size=num_targs)
        dec = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_targs))
        dec = np.rad2deg(dec-0.5*np.pi)
        tcoord = SkyCoord(ra=ra, dec=dec, unit='deg')
        pitch = tcoord.separation(sun).to_value("deg")
        good = (pitch > 46.2) & (pitch < 177.9)
        targets = np.append(targets, tcoord[good])

    targets = SkyCoord(targets[:num_targs])
    ra = targets.ra.to_value('deg')
    dec = targets.ra.to_value('deg')
    sun_ra = sun.ra.to_value('deg')
    sun_dec = sun.dec.to_value('deg')

    roll = [
        Ska.Sun.nominal_roll(r, d, sun_ra=sun_ra, sun_dec=sun_dec)
        for r, d in zip(ra, dec)
    ]
 
    eq = np.array([ra, dec, roll]).T
    q = Quat(equatorial=eq)

    return q


class RunFPTempModels:
    def __init__(self, tstart, tstop, model_spec=None):
        if model_spec is None:
            model_spec = get_xija_model_file('acisfp')
        self.ephem_table = ascii.read("ephem2023.dat")
        self.ephem_times = CxoTime(self.ephem_table["dates"].data).secs
        self.model_spec = model_spec
        self._get_orbits(tstart, tstop)
        self.num_orbits = self.orbits.size

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
        return self.ephem_times[idxs[0]:idxs[1]], solarephem, orbitephem

    def _get_orbits(self, tstart, tstop):
        from scipy.signal import find_peaks
        times, _, orbitephem = self._get_ephem(tstart, tstop)
        r = np.sqrt((orbitephem**2).sum(axis=0))
        idxs, _ = find_peaks(1.0/r)
        self.orbits = times[idxs]

    def calc_model(self, tstart, tstop, states):
        model = xija.ThermalModel("acisfp", start=tstart, stop=tstop,
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
        model.comp['dh_heater'].set_data(0.0, model.times)
        model.comp["fptemp"].set_data(-119.8, None)

        for i in range(1, 5):
            model.comp[f'aoattqt{i}'].set_data(states[f'q{i}'], state_times)

        for i, axis in enumerate("xyz"):
            model.comp[f'orbitephem0_{axis}'].set_data(orbitephem[i,:], ephem_times)

        model.comp['dpa_power'].set_data(0.0)
        model.comp['1cbat'].set_data(-53.0)
        model.comp['sim_px'].set_data(-120.0)

        model.make()
        model.calc()

        return model

    def run_models(self, exp_time=30.0, ntargs=100):
        import h5py
        exp_time *= 1000.0
        for i in range(0, self.num_orbits-1):
            times = np.arange(self.orbits[i], self.orbits[i+1], exp_time)
            f = h5py.File(f"acisfp_model_orbit_{i}.h5", "w")
            f.attrs['tstart'] = self.orbits[i]
            f.attrs['tstop'] = self.orbits[i+1]
            nobs = len(times)-1
            datestart = CxoTime(self.orbits[i]).date
            datestop = CxoTime(self.orbits[i+1]).date
            f.attrs['datestart'] = datestart
            f.attrs['datestop'] = datestop
            print(f"Running orbit from {datestart} to {datestop}.")
            for j in range(nobs):
                g = f.require_group(f"obs_{j}")
                g.attrs['tstart'] = times[j]
                g.attrs['tstop'] = times[j+1]
                g.attrs['datestart'] = CxoTime(times[j]).date
                g.attrs['datestop'] = CxoTime(times[j+1]).date
                q = generate_targets(times[j], ntargs)
                t = []
                mvals = []
                ephem_t = []
                ephem_x = []
                ephem_y = []
                ephem_z = []
                pitch = []
                g.create_dataset("q", data=q.q)
                for k in range(q.shape[0]):
                    states = make_states(times[j], times[j+1], 4, q.q[k,:])
                    model = self.calc_model(times[j], times[j+1], states)
                    t.append(model.times)
                    mvals.append(model.comp["fptemp"].mvals)
                    ephem_t.append(model.comp["orbitephem0_x"].times)
                    ephem_x.append(model.comp["orbitephem0_x"].dvals)
                    ephem_y.append(model.comp["orbitephem0_y"].dvals)
                    ephem_z.append(model.comp["orbitephem0_z"].dvals)
                    pitch.append(model.comp["pitch"].dvals)
                t = np.array(t)
                mvals = np.array(mvals)
                ephem_t = np.array(ephem_t)
                ephem_x = np.array(ephem_x)
                ephem_y = np.array(ephem_y)
                ephem_z = np.array(ephem_z)
                pitch = np.array(pitch)
                g.create_dataset("times", data=t)
                g.create_dataset("fptemp", data=mvals)
                g.create_dataset("ephem_t", data=ephem_t)
                g.create_dataset("ephem_x", data=ephem_x)
                g.create_dataset("ephem_y", data=ephem_y)
                g.create_dataset("ephem_z", data=ephem_z)
                g.create_dataset("pitch", data=pitch)

            f.flush()
            f.close()


if __name__ == "__main__":
    runner = RunFPTempModels("2019:100:00:00:00", "2019:200:00:00:00")
    runner.run_models()
