import acispy
from kadi.events import rad_zones
import numpy as np
import h5py
from cxotime import CxoTime
from fptemp_study import RunFPTempModels

runner = RunFPTempModels("2021:001:00:00:00", "2023:365:23:59:59",
                         fp_model_spec="acisfp_model_spec.json",
                         dpa_model_spec="dpa_model_spec.json")

ds = acispy.EngArchiveData("2020:001:00:00:00", "2020:365:23:59:59", 
                           [f"aoattqt{x}" for x in range(1, 5)])

rzs = rad_zones.filter("2020:001:00:00:00", "2020:365:23:59:59")

q2020 = []
t2020 = []
times = ds["aoattqt1"].times.value
for rz in rzs:
    idxs = (times >= rz.tstart) & (times <= rz.tstop)
    qq = np.array([ds["aoattqt1"].value[idxs],
                   ds["aoattqt2"].value[idxs],
                   ds["aoattqt3"].value[idxs],
                   ds["aoattqt4"].value[idxs]]).T
    t2020.append(times[idxs]-CxoTime(rz.perigee).secs)
    q2020.append(qq)


t = []
q = []
pad_time = 6000.0
current_year = "2021"
i = 0
for j in range(runner.num_orbits):
    tstart = runner.el_times[j, 0]-pad_time
    tstop = runner.el_times[j, 1]+pad_time
    per_time = runner.per_times[j]
    per_date = runner.per_dates[j]
    year, doy = per_date.split(":")[:2]
    if year != current_year:
        current_year = year
        i = 0
    print(current_year, per_date)
    if i >= len(t2020):
        times = None
        qq = None
    else:
        times, _, ephem = runner._get_ephem(tstart, tstop)
        tt = times-per_time
        qq = [np.interp(tt, t2020[i], q2020[i][:,j]) for j in range(0, 4)]
    t.append(times)
    q.append(np.array(qq).T)
    i += 1

with h5py.File("2020_atts.h5", "w") as f:
    for i in range(len(t)):
        g = f.create_group(f"orbit_{i}")
        if t[i] is not None:
            g.create_dataset("t", data=t[i])
            g.create_dataset("q", data=q[i])





