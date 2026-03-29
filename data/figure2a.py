# 
# figure2a.py
#
# D. Clarke
# 
import numpy as np
from common import Ls, jackSusc, d, MAINFOLD, Tfiles, Ts, injectTc
from latqcdtools.base.utilities import toNumpy
from latqcdtools.base.fileSystem import ls
from latqcdtools.base.readWrite import readTable
from latqcdtools.base.plotting import * 
from latqcdtools.statistics.statistics import * 
from latqcdtools.base.check import ignoreUnderflow

from latqcdtools.base.plotting import getColorGradient

Lcolors = getColorGradient(len(Ls))

ignoreUnderflow()
latexify()

injectTc()

if MAINFOLD=='Analysis_Scale_1':
    Tfiles, Ts = [], []
    temp = ls(f'{MAINFOLD}/50/reconstructionErrors/*.txt')
    for file in temp:
        Tfile = file.split('/')[3]
        Tfiles.append(Tfile)
        T = float(Tfile[:-4])
        Ts.append(T)
    Ts = np.array(Ts)

linestyles=['-','--',':','-.']

for iL,L in enumerate(Ls):

    V = L**d

    chiPs, chiPes = [],[]
    for Tfile in Tfiles:
        T = float(Tfile[:-4])
        err = readTable(f'{MAINFOLD}/{L}/reconstructionErrors/{Tfile}',usecols=(1,))
        p = 1 - err
        chiP, chiPe = jackSusc(p)
        chiPs.append(chiP)
        chiPes.append(chiPe)
    chiPs, chiPes = toNumpy(chiPs, chiPes)

    chiPs  = chiPs*(V/Ts)
    chiPes = chiPes*(V/Ts)

    istyle = iL % 4
    plot_lines(Ts,chiPs,chiPes,color=Lcolors[iL],label=f'{L}',linestyle=linestyles[istyle],marker=None)

set_params(xmin=4.44,xmax=4.55,xlabel='$T$',ylabel='$\\chi_{\\rm MSE}$')
saveFigure('figs/fig2a.pdf')
plt.show()