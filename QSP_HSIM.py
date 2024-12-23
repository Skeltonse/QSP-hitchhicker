"""
DEMONSTRATES QSP PARAMETER FINDER AND SIMULATION
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time 
import tikzplotlib


'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import parameter_finder as pf
import solvers.Wilson_method as wm
from simulators.projector_calcs import BUILD_PLIST, Ep_CALL, Ep_PLOT, SU2_CHECK, UNIFY_PLIST
from simulators.LQSP_sim import COMPLEX_QSP_SIM, COMPLEX_QSP_SIM2, QSP_MMNT
import simulators.matrix_fcns as mf

from HS_benchmark import get_coeffs, HS_FCN_CHECK

'''SPECIFIED BY THE USER'''
epsi=float(10**(-12))
t=30
inst_subnorm=2
ifsave=True
pathname="QSP_HSIM.py"

'''DEFINE UNIVERSAL VARIABLES'''
filename="hsim_coeffs_epsi_" + "1.0e-14" + "_t_" + str(t) 
inst_tol=10**(-2)
H=mf.random_hermitian_matrix(8, 0.7)

'''DEFINE THE FIGURE AND DOMAIN'''
pts=500
theta=np.linspace(-np.pi,np.pi,pts)
xdata=np.cos(theta)

'''DEFINE PATHS FOR FILES'''
current_path=os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
save_path=os.path.join(coeff_path,"benchmark_data")
save_path = os.path.normpath(save_path)

czlist, szlist, n=get_coeffs(filename)

'''INITIAL CHECKS ON COEFF LISTS; OPTIONAL''' 
HS_FCN_CHECK(czlist, szlist, n, t, theta, xdata, inst_tol=inst_tol, subnorm=inst_subnorm, plots=True)

'''BEGIN FINDING THE SOLUTION: BUILD $\mathcal{F}(z)$ and solve for $c(z), d(z)$'''
Plist, Qlist, E0, a, b, c, d, n, tDict=pf.PARAMETER_FIND(czlist, -1j*szlist, n, theta, epsi=inst_tol,  plots=True)

def PLOT_LQSPSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=False, ):
    '''BUILD THE COMPLEX THE QSP ORACLE'''
    U, evals, evecs, Hevals=mf.UNITARY_BUILD(H, return_evals=True)

    '''RUN THE QSP CIRCUIT'''
    UPhi=COMPLEX_QSP_SIM(U, Plist, Qlist, E0)
    
    '''EXTRACT THE CORRECT EIGENVALUES'''
    QSPl=[]
    FCNl=[]
    for l in range(0, len(H)):
        QSPl.append(QSP_MMNT(UPhi,  evecs[:, l][:, np.newaxis])[0, 0])
        FCNl.append(np.exp(1j*Hevals[l]*t)/2)

    QSP=np.array(QSPl)
    FCN=np.array(FCNl)

    indices=np.argsort(Hevals)
    Hevals=Hevals[indices]
    QSP=QSP[indices]
    FCN=FCN[indices]
    fig, axes = plt.subplots(1, figsize=(12, 6))
    axes.scatter(Hevals, np.imag(QSP),color='blue', marker='1',label=r'$\langle{\lambda}|U_{\Phi}|{\lambda}\rangle_{imag}$')
    axes.plot(Hevals, np.imag(FCN), color='orange', marker='.', label=r'$\mathcal{A}_{imag}$')
    axes.scatter(Hevals, np.real(QSP), color='green',marker='1',label=r'$\langle{\lambda}|U_{\Phi}|{\lambda}\rangle_{real}$')
    axes.plot(Hevals, np.real(FCN),  color='red', marker=".", label=r'$\mathcal{A}_{real}$')
    plt.legend()
    plt.title(r"Expectation values of QSP circuit vs $f(\lambda)$")
    if ifsave==True:
        plt.savefig(save_path+"QSP_HS_for_t="+str(t)+".png")
    else:
        plt.show()
    return

def PLOT_QSPSIM_WITH_DATA(Plist, Qlist, E0, data, a, b, ifsave=False, withcomp=False):
    fig, axes = plt.subplots(1, figsize=(12, 8))
    
    Ep_PLOT(Plist, Qlist, E0, n, a, b, data,  ax=axes, )
    if withcomp==True:
        axes.scatter(data, np.real(lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+ 1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))))
        axes.scatter(data,np.imag(lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+ 1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))))

    if ifsave==True:
        plt.savefig(save_path+"QSP_HS_for_t="+str(t)+".png")
    else:
        plt.show()
    return


PLOT_LQSPSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=False, )
# PLOT_QSPSIM_WITH_DATA(Plist, Qlist, E0, theta, a, b)
