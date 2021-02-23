import matplotlib.pyplot as plt
from Ska.Matplotlib import plot_cxctime
from fptemp_study import RunFPTempModels

runner = RunFPTempModels("2021:001:00:00:00", "2023:365:23:59:59")

fig = plt.figure(figsize=(10,10))
plt.rc("axes", linewidth=2)
plt.rc("font", size=18)
_, _, ax = plot_cxctime(runner.per_dates, runner.a_perigee, fig=fig, lw=2, color="C1")
ax.tick_params(width=2, length=6)
ax.tick_params(which='minor', width=2, length=4)
ax.set_xlabel("Date")
ax.set_ylabel("Altitude (km)")
fig.savefig("perigee_vs_time.png")

