import h5py
from fptemp_study import RunFPTempModels
from cxotime import CxoTime
import acispy

runner = RunFPTempModels("2021:001:00:00:00", "2023:365:23:59:59")

asc_start = CxoTime(CxoTime(runner.el_dates[-139][0]).secs-6.0e3).date
asc_stop = runner.el_dates[-139][0]
perigee_date = runner.per_dates[-1]
dsc_start = runner.el_dates[-139][1]
dsc_stop = CxoTime(CxoTime(runner.el_dates[-139][1]).secs+6.0e3).date

fn = "data/acisfp_model_perigee_2023_363_2023_363_m109.h5"

which = 11

with h5py.File(fn, "r") as f:
    dates = CxoTime(f['times'][which]).date
    temp = f['fptemp'][which,:]
    fep_count = f['esa'][which,:]
    dp = acispy.CustomDatePlot(dates, temp)
    dp.plot_right(dates, fep_count, drawstyle='steps')
    dp.set_ylabel("FPTEMP_11 ($^\circ$C)")
    dp.add_vline(asc_start, lw=2, ls='--')
    dp.add_vline(asc_stop, lw=2, ls='--')
    dp.add_vline(perigee_date, lw=2, ls='--')
    dp.add_vline(dsc_start, lw=2, ls='--')
    dp.add_vline(dsc_stop, lw=2, ls='--')
    dp.set_ylabel2("Earth Solid Angle (sr)")
    dp.savefig("fptemp_example.png")

with h5py.File(fn, "r") as f:
    dates = CxoTime(f['times'][which]).date
    temp = f['1dpamzt'][which,:]
    print(f['pitch'][which,0])
    fep_count = f['fep_count'][which,:]
    dp = acispy.CustomDatePlot(dates, temp)
    dp.plot_right(dates, fep_count, drawstyle='steps')
    dp.set_ylabel("1DPAMZT ($^\circ$C)")
    dp.add_vline(asc_start, lw=2, ls='--')
    dp.add_vline(asc_stop, lw=2, ls='--')
    dp.add_vline(perigee_date, lw=2, ls='--')
    dp.add_vline(dsc_start, lw=2, ls='--')
    dp.add_vline(dsc_stop, lw=2, ls='--')
    dp.set_ylabel2("FEP Count")
    dp.savefig("1dpamzt_example.png")


