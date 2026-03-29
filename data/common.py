from latqcdtools.statistics.jackknife import jackknife
from latqcdtools.base.utilities import find_nearest_idx
from latqcdtools.base.fileSystem import ls
from latqcdtools.physics.statisticalPhysics import Z2_3d
from latqcdtools.base.plotting import plt, plot_vspan, plot_hspan
import latqcdtools.base.logger as logger
import numpy as np
import gvar as gv


univ   = Z2_3d()
TcLIT, TcLITe = gv.mean(univ.Tc), gv.sdev(univ.Tc)

def injectTc(ax=plt,orientation='vertical'):
    logger.info('Injecting Tc')
    if orientation=='vertical':
        plot_vspan(TcLIT-TcLITe,TcLIT+TcLITe,color='grey',label='$T_{\\rm c}$',alpha=0.3,ax=ax)
    if orientation=='horizontal':
        plot_hspan(TcLIT-TcLITe,TcLIT+TcLITe,color='grey',label='$T_{\\rm c}$',alpha=0.3,ax=ax)


Ls = [50,60,70,80,90,100,110,120,130]
NBLOCKS = 40
d=3

MAINFOLD='Analysis_Scale_1'

Tfiles, Ts = [], []
temp = ls(f'Analysis_Scale_1/50/reconstructionErrors/*.txt')
for file in temp:
    Tfile = file.split('/')[3]
    Tfiles.append(Tfile)
    T = float(Tfile[:-4])
    Ts.append(T)
Ts = np.array(Ts)
if len(Ts)==0:
    logger.TBError('Mess up with Ts')

Tpairs, Tfilepairs = [], []
for iT in range(len(Ts)//2):
    Tpair=(Ts[iT],Ts[-iT-1])
    Tfilepair=(Tfiles[iT],Ts[-iT-1])
    Tpairs.append(Tpair)
    Tfilepairs.append(Tfilepair)

def roughTc(Tin,chi,chie):
    chimax = np.max(chi)
    iTc = find_nearest_idx(chi,chimax)
    indices = [iTc]
    for i in range(4):
        chileft   = chi[iTc-i]
        chilefte  = chie[iTc-i]
        chiright  = chi[iTc+i]
        chirighte = chie[iTc+i]
        if chileft-chilefte<chimax<chileft+chilefte:
            indices.append(find_nearest_idx(chi,chileft))
        if chiright-chirighte<chimax<chiright+chirighte:
            indices.append(find_nearest_idx(chi,chiright))
    Tcs = []
    for iT in indices:
        Tcs.append(Tin[iT])
    Tcs = np.array(Tcs)
    Tce = (np.max(Tcs)-np.min(Tcs))/2
    Tcm = (np.max(Tcs)+np.min(Tcs))/2
    return Tcm, Tce, Tin[iTc]

def jack(data):
    return jackknife(np.mean,data,NBLOCKS)

def susc(data):
    return np.mean(data**2)-np.mean(data)**2

def jackSusc(data):
    return jackknife(susc,data,NBLOCKS)

