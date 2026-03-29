# 
# figure1b.py
# 
# D. Clarke
# 
from common import Tfiles, Ts, jackSusc, d, injectTc, plt
from latqcdtools.base.utilities import toNumpy
from latqcdtools.base.readWrite import readTable
from latqcdtools.base.plotting import  latexify, plot_lines, saveFigure, set_params, plot_vspan
from latqcdtools.base.check import ignoreUnderflow

ignoreUnderflow()
latexify()

fig, ax = plt.subplots()
inset = fig.add_axes([0.57, 0.3, 0.3, 0.3])  # [left, bottom, width, height]

for iL,L in enumerate([50,130]):
    
    eps = 0.0001
    
    Tclip = Ts 
    
    V = L**d

    Ps, Pes, P2s, P2es = [],[],[],[]
    for Tfile in Tfiles:

        err  = readTable(f'Analysis_Scale_1/{L}/reconstructionErrors/{Tfile}',usecols=(1,))
        err2 = readTable(f'Analysis_Scale_2/{L}/reconstructionErrors/{Tfile}',usecols=(1,))

        p1m, p1e = jackSusc(err)
        p2m, p2e = jackSusc(err2)
        Ps.append(  p1m)
        P2s.append( p2m)
        Pes.append( p1e)
        P2es.append(p2e)
    Ps, Pes, P2s, P2es = toNumpy(Ps, Pes, P2s, P2es) 

    Ps   = (V/Tclip)*Ps 
    Pes  = (V/Tclip)*Pes 
    P2s  = (V/Tclip)*P2s 
    P2es = (V/Tclip)*P2es 

    if iL==0:
        plot_lines(Tclip    ,Ps ,Pes ,label='CAE1',linestyle='solid',color='purple',marker=None,ax=ax)
        plot_lines(Tclip+eps,P2s,P2es,label='CAE2',linestyle='solid',color='blue'  ,marker=None,ax=ax)
        plot_lines(Tclip    ,Ps ,Pes ,linestyle='solid',color='purple',marker=None,ax=inset)
        plot_lines(Tclip+eps,P2s,P2es,linestyle='solid',color='blue'  ,marker=None,ax=inset)

injectTc(ax=ax,orientation='vertical')

plot_vspan(4.4790-4e-3              ,4.4790+4e-3              ,color='purple',alpha=0.3,ax=inset)
plot_vspan(4.48275909-2.59215644e-03,4.48275909+2.59215644e-03,color='blue'  ,alpha=0.3,ax=inset)

set_params(xlabel='$T$',ylabel='$\\chi_{\\rm MSE}$',ax=ax)
set_params(xmin=4.4,xmax=4.55,ax=inset)

saveFigure(f'figs/fig1b.pdf')
plt.show()

