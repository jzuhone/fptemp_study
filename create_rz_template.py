from kadi.commands import get_cmds_from_backstop
from cxotime import CxoTime
import numpy as np

bs_cmds = get_cmds_from_backstop("/data/acis/LoadReviews/2020/DEC2120/oflsb/CR356_0104.backstop")

tstart = CxoTime("2020:356:18:11:00.839").secs
tstop = CxoTime("2020:357:05:28:38.615").secs

rz_cmds = bs_cmds[(bs_cmds["time"] >= tstart) & (bs_cmds["time"] <= tstop)]
rz_cmds.remove_column("params")
rz_cmds['params'] = np.array(['']*len(rz_cmds))
rz_cmds["idx"].dtype = int
rz_cmds.write("rz_template.dat", format='ascii.commented_header', overwrite=True)

