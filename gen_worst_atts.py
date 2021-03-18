from fptemp_study import RunFPTempModels
from gen_targets import generate_worst_targets
import numpy as np
from tqdm.auto import trange
import h5py

runner = RunFPTempModels("2021:001:00:00:00", "2023:365:23:59:59",
                         fp_model_spec="acisfp_model_spec.json",
                         dpa_model_spec="dpa_model_spec.json")

t = []
q = []
pad_time = 6000.0
for j in trange(runner.num_orbits):
    tstart = runner.el_times[j, 0]-pad_time
    tstop = runner.el_times[j, 1]+pad_time
    times, _, ephem = runner._get_ephem(tstart, tstop)
    t.append(times)
    q.append(generate_worst_targets(times, ephem, num_targs=1000))

t = np.array(t)
q = np.array(q)

with h5py.File("worst_atts.h5", "w") as f:
    for i in range(len(t)):
        g = f.create_group(f"orbit_{i}")
        g.create_dataset("t", data=t[i])
        g.create_dataset("q", data=q[i])
