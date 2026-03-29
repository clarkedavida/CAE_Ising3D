# 
# figure2b.py 
# 
# D. Clarke
# 
from common import Ls, jack, d, MAINFOLD
from latqcdtools.base.utilities import toNumpy
from latqcdtools.base.fileSystem import ls
from latqcdtools.base.readWrite import readTable
from latqcdtools.base.plotting import set_params, latexify, plt, \
    getColorGradient, saveFigure, plot_lines
import numpy as np

latexify()

colorsL = getColorGradient(len(Ls))

for iL, L in enumerate(Ls):

    V = L**d
    files = ls(f'{MAINFOLD}/{L}/reconstructionErrors/*.txt')
    Ts, strTs, Ps, Pes, chiPs, chiPes, Ms, Mes, chis, chies = [],[],[],[],[],[],[],[],[],[]

    for file in files:

        strT = file.split('/')[3][:-4]
        strTs.append(strT)
        T = float(strT)

        try:        
            M = readTable(f'{MAINFOLD}/{L}/Observables_per_config/all_conf_at_T{strT[:-2]}.txt',usecols=(1))
            P = readTable(file,usecols=(1))
        except:
            continue
 
        P = 1-P

        Pm, Pe = jack(P) 
        Ps.append(Pm)
        Ts.append(T)
        Pes.append(Pe)

        M = np.abs(M)
        Mm, Me = jack(M) 
        Ms.append(Mm)
        Mes.append(Me)

    Ts, strTs, Ps, Pes, Ms, Me  = toNumpy(Ts, strTs, Ps, Pes, Ms, Mes)

    plot_lines(Ms,Ps,marker=None,color=colorsL[iL],label=f'{L}')


set_params(xlabel='$\\ev{|m|}$',ylabel='$1-\\ev{\\rm MSE}$',ymax=1.2)

saveFigure(f'figs/fig2b.pdf')

plt.show()
