from fptemp_study import RunFPTempModels
from gen_targets import generate_constant_targets
from tqdm.auto import trange
import h5py

runner = RunFPTempModels("2021:001:00:00:00", "2023:365:23:59:59",
                         fp_model_spec="acisfp_model_spec.json",
                         dpa_model_spec="dpa_model_spec.json")

q = []
pad_time = 6000.0
for j in trange(runner.num_orbits):
    tstart = runner.el_times[j, 0]-pad_time
    q.append(generate_constant_targets(tstart, 1000))

with h5py.File("const_atts.h5", "w") as f:
    for i in range(len(q)):
        g = f.create_group(f"orbit_{i}")
        g.create_dataset("q", data=q[i])
