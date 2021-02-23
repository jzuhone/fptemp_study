import numpy as np
from astropy.io import ascii
import xija
from astropy.constants import R_earth
from cxotime import CxoTime
import h5py
from collections import defaultdict
import Ska.Numpy
from gen_targets import generate_constant_targets

R_e = R_earth.to_value("km")

msid = {"dpa": "1dpamzt",
        "acisfp": "fptemp"}


def make_radzone_states(datestart, datestop, eefdate, xefdate, q,
                        pow2atime=None):
    from kadi.commands import CommandTable, states
    tstart = float(CxoTime(datestart).secs)
    tstop = float(CxoTime(datestop).secs)
    eeftime = float(CxoTime(eefdate).secs)
    xeftime = float(CxoTime(xefdate).secs)
    rz_cmds = CommandTable.read("rz_template.dat",
                                format='ascii.commented_header')
    # Get time differences
    dt = np.diff(rz_cmds["time"])
    # Make a new time array and reconstruct the times
    new_time = rz_cmds["time"].copy()
    # first stop science
    new_time[0] = tstart
    # second stop science
    new_time[1] = new_time[0]+dt[0]
    # power-down all FEPs and video boards
    new_time[2] = new_time[1]+dt[1]
    # power up FEPs and video boards
    new_time[3] = new_time[2]+dt[2]
    # start science
    new_time[4] = new_time[3]+dt[3]
    # stop science
    new_time[5] = eeftime
    # stop science again
    new_time[6] = new_time[5]+dt[5]
    # power-down all FEPs and video boards
    new_time[7] = new_time[6]+dt[6]
    # power up three FEPs
    if pow2atime is None:
        new_time[8] = new_time[7] + 3.0*3600.0
    else:
        new_time[8] = pow2atime
    # first stop science
    new_time[9] = xeftime
    # second stop science
    new_time[10] = new_time[9]+dt[9]
    # power-down all FEPs and video boards
    new_time[11] = new_time[10]+dt[10]
    # power up FEPs and video boards
    new_time[12] = new_time[11]+dt[11]
    # start science
    new_time[13] = new_time[12]+dt[12]
    # stop science
    new_time[14] = tstop
    rz_cmds["time"] = new_time
    rz_cmds["date"] = CxoTime(new_time).date
    continuity = {
        'ccd_count': 4,
        'fep_count': 4,
        'clocking': 0,
        'vid_board': 1,
        'simpos': -99616.0,
        'q1': q[0],
        'q2': q[1],
        'q3': q[2],
        'q4': q[3]
    }
    continuity['__dates__'] = {k: CxoTime(tstart-60.0).date for k in continuity}
    st = states.get_states(cmds=rz_cmds, continuity=continuity)
    return st.as_array()


def make_constant_states(start, stop, ccd_count, q, 
                         fep_count=None, clocking=1,
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


class RunFPTempModels:
    def __init__(self, tstart=None, tstop=None, fp_model_spec=None,
                 dpa_model_spec=None):
        if fp_model_spec is None:
            fp_model_spec = 'acisfp_model_spec.json'
        if dpa_model_spec is None:
            dpa_model_spec = 'dpa_model_spec.json'
        self.ephem_table = ascii.read("ephem2023.dat")
        self.ephem_times = CxoTime(self.ephem_table["dates"].data).secs
        if tstart is None:
            tstart = self.ephem_times[0]
        if tstop is None:
            tstop = self.ephem_times[-1]
        self.fp_model_spec = fp_model_spec
        self.dpa_model_spec = dpa_model_spec
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

    def calc_model(self, name, tstart, tstop, states, model_spec, T_init=-119.8):
        model = xija.ThermalModel(name, start=tstart, stop=tstop,
                                  model_spec=model_spec)
        ephem_times, solarephem, orbitephem = self._get_ephem(tstart, tstop)
        state_times = np.array([states['tstart'], states['tstop']])
        model.comp['sim_z'].set_data(states['simpos'], state_times)
        model.comp['eclipse'].set_data(False)
        for k in ('ccd_count', 'fep_count', 'vid_board', 'clocking'):
            model.comp[k].set_data(states[k], state_times)
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

    def get_target(self, which_targs, orbit_num):
        if which_targs == "const":
            with h5py.File("const_atts.h5", "r") as f:
                g = f[f"orbit_{orbit_num}"]
                q = g["q"]
        else:
            raise NotImplementedError
        return q

    def run_perigees(self, which_targs, T_init=-119.8):
        pad_time = 6000.0
        for i in range(0, self.num_orbits):
            tstart = self.el_times[i, 0]-pad_time
            tstop = self.el_times[i, 1]+pad_time
            datestart = CxoTime(tstart).date
            datestop = CxoTime(tstop).date
            eefdate = self.el_dates[i, 0]
            xefdate = self.el_dates[i, 1]
            b = datestart[:8].replace(":", "_")
            e = datestop[:8].replace(":", "_")
            T = f"m{np.abs(np.round(T_init).astype('int'))}"
            f = h5py.File(f"data/acisfp_model_perigee_{b}_{e}_{T}.h5", "w")
            f.attrs['tstart'] = tstart
            f.attrs['tstop'] = tstop
            f.attrs['datestart'] = datestart
            f.attrs['datestop'] = datestop
            q = self.get_target(which_targs, i)
            f.create_dataset("q", data=q)
            output_fields = ["times", "fptemp", "1dpamzt", "ephem_t", 
                             "ephem_x", "ephem_y", "ephem_z",
                             "pitch", "roll", "esa", "ccd_count",
                             "fep_count", "clocking"]
            y = defaultdict(list)
            for iq in range(q.shape[0]):
                states = make_radzone_states(datestart, datestop, eefdate,
                                             xefdate, q[iq, :])
                dpa_model = self.calc_model("dpa", tstart, tstop, states,
                                            self.dpa_model_spec, T_init=20.0)
                idxs = (dpa_model.comp["1dpamzt"].mvals < 12.0) & \
                       (dpa_model.comp["fep_count"].dvals == 0)
                if idxs.sum() > 0:
                    pow2atime = dpa_model.times[idxs][0]
                    states = make_radzone_states(datestart, datestop, eefdate,
                                                 xefdate, q.q[iq, :],
                                                 pow2atime=pow2atime)
                    dpa_model = self.calc_model("dpa", tstart, tstop, states,
                                                self.dpa_model_spec, T_init=20.0)
                fp_model = self.calc_model("acisfp", tstart, tstop, states,
                                           self.fp_model_spec, T_init=T_init)
                for k in output_fields:
                    if k == "times":
                        v = fp_model.times
                    elif k == "fptemp":
                        v = fp_model.comp["fptemp"].mvals
                    elif k == "1dpamzt":
                        v = dpa_model.comp["1dpamzt"].mvals
                    elif k.startswith("ephem"):
                        if k.endswith("t"):
                            v = fp_model.comp["orbitephem0_x"].times
                        else:
                            v = fp_model.comp[f"orbitephem0_{k[-1]}"].dvals
                    elif k == "esa":
                        v = fp_model.comp["earthheat__fptemp"].dvals
                    else:
                        v = fp_model.comp[k].dvals
                    y[k].append(v)
            for k in output_fields:
                f.create_dataset(k, data=np.array(y[k]))
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
            f = h5py.File(f"data/acisfp_model_orbit_{b}_{e}_{T}.h5", "w")
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
                q = generate_constant_targets(times[j], ntargs)
                g.create_dataset("q", data=q)
                output_fields = ["times", "fptemp", "ephem_t",
                                 "ephem_x", "ephem_y", "ephem_z",
                                 "pitch", "roll", "esa"]
                y = defaultdict(list)
                for iq in range(q.shape[0]):
                    states = make_constant_states(times[j], times[j]+exp_time, 
                                                  4, q[iq, :])
                    model = self.calc_model("acisfp", times[j], times[j]+exp_time, 
                                            states, self.fp_model_spec, T_init=T_init)
                    for k in output_fields:
                        if k == "times":
                            v = model.times
                        elif k == "fptemp":
                            v = model.comp["fptemp"].mvals
                        elif k.startswith("ephem"):
                            if k.endswith("t"):
                                v = model.comp["orbitephem0_x"].times
                            else:
                                v = model.comp[f"orbitephem0_{k[-1]}"].dvals
                        elif k == "esa":
                            v = model.comp["earthheat__fptemp"].dvals
                        else:
                            v = model.comp[k].dvals
                        y[k].append(v)
                for k in output_fields:
                    g.create_dataset(k, data=np.array(y[k]))

            f.flush()
            f.close()


if __name__ == "__main__":
    runner = RunFPTempModels("2021:001:00:00:00", "2023:365:23:59:59",
                             fp_model_spec="acisfp_model_spec.json",
                             dpa_model_spec="dpa_model_spec.json")
    #runner.run_observations()
    runner.run_perigees(T_init=-109.0)