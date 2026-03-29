# 
# figure1a.py 
# 
# D. Clarke
# 
from common import Tfiles, Ts, jack, injectTc, plt
from latqcdtools.base.utilities import toNumpy
from latqcdtools.base.readWrite import readTable
from latqcdtools.base.plotting import  latexify, plot_lines, saveFigure, set_params
from latqcdtools.base.check import ignoreUnderflow

ignoreUnderflow()
latexify()

fig, ax = plt.subplots()
inset = fig.add_axes([0.57, 0.3, 0.3, 0.3])  # [left, bottom, width, height]

for iL,L in enumerate([50,130]):
    
    eps = 0.0001
    
    Tclip = Ts 

    Ps, Pes, P2s, P2es = [],[],[],[]
    for Tfile in Tfiles:

        err  = readTable(f'Analysis_Scale_1/{L}/reconstructionErrors/{Tfile}',usecols=(1,))
        err2 = readTable(f'Analysis_Scale_2/{L}/reconstructionErrors/{Tfile}',usecols=(1,))

        p1m, p1e = jack(err)
        p2m, p2e = jack(err2)
        Ps.append(  p1m)
        P2s.append( p2m)
        Pes.append( p1e)
        P2es.append(p2e)
    Ps, Pes, P2s, P2es = toNumpy(Ps, Pes, P2s, P2es) 

    if iL==0:
        plot_lines(Tclip    ,Ps ,Pes ,label='CAE1',linestyle='solid',color='purple',marker=None,ax=ax)
        plot_lines(Tclip+eps,P2s,P2es,label='CAE2',linestyle='solid',color='blue'  ,marker=None,ax=ax)
        plot_lines(Tclip    ,Ps ,Pes ,linestyle='solid',color='purple',marker=None,ax=inset)
        plot_lines(Tclip+eps,P2s,P2es,linestyle='solid',color='blue'  ,marker=None,ax=inset)
    else:
        plot_lines(Tclip+2*eps,Ps ,Pes ,linestyle='dotted',color='purple',marker=None,ax=inset)
        plot_lines(Tclip+3*eps,P2s,P2es,linestyle='dotted',color='blue',marker=None,ax=inset)

injectTc(ax=ax,orientation='vertical')

set_params(xlabel='$T$',ylabel='$\\ev{\\rm MSE}$',legendpos='center left',ax=ax)
set_params(xmin=4.4,xmax=4.55,ymin=0.7,ymax=1.1,ax=inset)

saveFigure(f'figs/fig1a.pdf')
plt.show()

