# 
# figure3.py                                                               
# 
# D. Clarke
# 

import numpy as np
from latqcdtools.base.readWrite import readTable
from latqcdtools.base.plotting import * 
from latqcdtools.base.printErrorBars import get_err_str
import latqcdtools.base.logger as logger
from latqcdtools.statistics.statistics import gaudif
from latqcdtools.statistics.fitting import Fitter
from latqcdtools.physics.statisticalPhysics import Z2_3d
from common import *

import gvar as gv

latexify()

univ   = Z2_3d()
TcLIT, TcLITe = gv.mean(univ.Tc), gv.sdev(univ.Tc)

FINITEVCORRECTION=False

univ.exponentSummary()

def linearFit(x,p):
    return p[0] + p[1]*x

def nuTcFit(x,p):
    return p[1] + p[2]*x**(1/p[0]) 

L, TcL, TcLerr, chiL, chiLerr = readTable('Perr_ALGthorough_TMIN0_TMAX6.txt')

xdata=1/L
plot_dots(1/L,TcL,TcLerr,label='$T_{\\rm c}^{\\rm CAE}(L)$')

fit = Fitter(nuTcFit,xdata,TcL,TcLerr)
res, reserr, chidof = fit.try_fit(algorithms=['curve_fit'],start_params=[1,TcLIT,-1])
fit.plot_fit(domain=(1e-6,np.max(xdata)))

nu=res[0]
nue=reserr[0]
Tc=res[1]
Tce=reserr[1]

logger.info('    nu =',get_err_str(nu,nue))
logger.info('    Tc =',get_err_str(Tc,Tce))
logger.info('chidof =',round(chidof,2))
logger.info('  q_nu =',round(gaudif(nu,nue,gv.mean(univ.nu),gv.sdev(univ.nu)),3))
logger.info('  q_Tc =',round(gaudif(Tc,Tce,TcLIT,TcLITe),3))


plot_dots(xdata=[0],ydata=[Tc],yedata=[Tce],color='blue',marker='*',label='$T_{\\rm c}^{\\rm CAE}$',markerfill=True)

injectTc(ax=plt,orientation='horizontal')
set_params(xlabel='$1/L$',ylabel='$T$')

saveFigure('figs/fig3.pdf')
plt.show()
