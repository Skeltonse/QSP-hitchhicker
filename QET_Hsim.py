"""
Used to be the main HS file, now it has a similar function to QET_test_script.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time 

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{braket}')
#matplotlib.verbose.level = 'debug-annoying'


'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import parameter_finder as pf
from functions.matrix_inverse import CHEBY_INV_COEFF_ARRAY
import solvers.Wilson_method as wm
from simulators.projector_calcs import BUILD_PLIST, Ep_CALL, Ep_PLOT, SU2_CHECK, UNIFY_PLIST
from simulators.angle_calcs import PROJ_TO_ANGLE, Wx_TO_R, W_PLOT, PAULI_CHECK, HAAHR_PLOT, HAAHW_PLOT
from simulators.qet_sim import COMPLEX_QET_SIM, COMPLEX_QET_SIM2, QET_MMNT

from scipy.linalg import expm
import simulators.matrix_fcns as mf
from simulators.unitary_calcs import GQSP_SIM, COMPUTE_R, GQSP_LSIM, SENSIBLE_MATRIX

from HS_benchmark import get_coeffs, HS_FCN_CHECK

'''SPECIFIED BY THE USER'''
epsi=float(10**(-14))
t=50
ifsave=True
device='pc'
pathname="QET_Hsim.py"
defconv='ad'

'''DEFINE UNIVERSAL VARIABLES'''
filename="hsim_coeffs_epsi_" + "1.0e-14" + "_t_" + str(t) 
inst_tol=10**(-12)
H=mf.random_hermitian_matrix(8, 0.7)

'''DEFINE THE FIGURE AND DOMAIN'''

pts=500
xdata=np.linspace(-1, 1, pts)
data=np.arccos(xdata)

defconv='ad'

'''DEFINE PATHS FOR FILES'''
if device=='mac':
    current_path = os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"benchmark_data/")
else:
    current_path = os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"\benchmark_data")


czlist, szlist, n=get_coeffs(filename)


'''INITIAL CHECKS ON COEFF LISTS; SHOULD EVENTUALLY BE OPTIONAL''' 
#HS_FCN_CHECK(czlist, -1j*czlist, n, t, data, xdata, subnorm=2, plots=True)

'''BEGIN FINDING THE SOLUTION: BUILD $\mathcal{F}(z)$ and solve for $c(z), d(z)$'''
Plist, Qlist, E0, a, b, c, d, n, tDict=pf.PARAMETER_FIND(czlist, -1j*szlist, n, data, inst_tol, defconv, plots=True)

def PLOT_GQSP_WITH_GIVEN_H(a, b, c, d, H):
    P=a+1j*b
    Q=1j*c+d

    tlist, plist, lamb=COMPUTE_R(P, Q)
    U, evals, evecs, Hevals=mf.UNITARY_BUILD(H, return_evals=True)

    UPhi=GQSP_SIM(tlist, plist, lamb, U, convent=np.array([[1], [0]]))

    GQSP=np.zeros([len(evals)], dtype=complex)
    for l in range(0, len(H)):
        GQSP[l]=QET_MMNT(UPhi,  evecs[:, l][:, np.newaxis])[0, 0]

    fig, axes = plt.subplots(1, figsize=(12, 6))
    fcn2=(np.exp(1j*data)**(n))*(lpf.LAUR_POLY_BUILD(a, n,  np.exp(1j*data))+1j*lpf.LAUR_POLY_BUILD(b, n,  np.exp(1j*data)))

    axes.scatter(Hevals, np.imag(GQSP),color='blue', marker='1',label=r'Imag G-QSP')
    axes.scatter(Hevals, np.real(GQSP), color='green',marker='1',label=r'Real G-QSP')
    plt.plot(xdata, np.real(fcn2))
    plt.plot(xdata, np.imag(fcn2))
    plt.show()
    return


#tlist, plist, lamb=COMPUTE_R(P, Q)
#U, evals, evecs, Hevals=mf.UNITARY_BUILD(H, return_evals=True)

#UPhi=GQSP_LSIM(tlist, plist, lamb, U,n, convent=np.array([[1], [1]])/np.sqrt(2))

#GQSP=np.zeros([len(evals)], dtype=complex)
#for l in range(0, len(H)):
#    GQSP[l]=QET_MMNT(UPhi,  evecs[:, l][:, np.newaxis])[0, 0]

#fig, axes = plt.subplots(1, figsize=(12, 6))
#fcn2=(lpf.LAUR_POLY_BUILD(a, n,  np.exp(1j*data))+1j*np.exp(1j*data)*lpf.LAUR_POLY_BUILD(b, n,  np.exp(1j*data)**2))

#axes.scatter(Hevals, np.imag(GQSP),color='blue', marker='1',label=r'Imag G-QSP')
#axes.scatter(Hevals, np.real(GQSP), color='green',marker='1',label=r'Real G-QSP')
#plt.plot(xdata, np.real(fcn2))
#plt.plot(xdata, np.imag(fcn2))
#plt.show()

def PLOT_QETSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=False, ):
    '''BUILD THE COMPLEX THE QSP ORACLE'''
    U, evals, evecs, Hevals=mf.UNITARY_BUILD(H, return_evals=True)

    '''RUN THE QSP CIRCUIT'''
    UPhi=COMPLEX_QET_SIM(U, Plist, Qlist, E0)
    
    '''EXTRACT THE CORRECT EIGENVALUES'''
    QETl=[]
    FCNl=[]
    for l in range(0, len(H)):
        QETl.append(QET_MMNT(UPhi,  evecs[:, l][:, np.newaxis])[0, 0])
        FCNl.append(np.exp(1j*Hevals[l]*t)/2)

    QET=np.array(QETl)
    FCN=np.array(FCNl)
    fig, axes = plt.subplots(1, figsize=(12, 6))
    axes.scatter(Hevals, np.imag(QET),color='blue', marker='1',label=r'$\bra{\lambda}U_{\Phi}\ket{\lambda}_{imag}$')
    axes.scatter(Hevals, np.imag(FCN), color='orange', marker='.', label=r'$\mathcal{A}_{imag}$')
    axes.scatter(Hevals, np.real(QET), color='green',marker='1',label=r'$\bra{\lambda}U_{\Phi}\ket{\lambda}_{real}$')
    axes.scatter(Hevals, np.real(FCN),  color='red', marker=".", label=r'$\mathcal{A}_{real}$')


    if ifsave==True:
        plt.savefig(save_path+"QET_HS_for_t="+str(t)+".png")
    else:
        plt.show()
    return

def PLOT_REALQSPSIM_WITH_DATA(Plist, Qlist, E0, n, epsi, data,  ifsave=False, ):
    fig, axes = plt.subplots(2, figsize=(12, 8))
    '''COMPUTE PARAMS FOR NEW QSP CIRCUIT'''
    print(np.trace(Plist[:, :, 12]@np.array([[0, 1], [1, 0]])))
    philist=PROJ_TO_ANGLE(Plist, E0, n, recip=defconv,tol=epsi)
    #philistr=Wx_TO_R(philist)
    '''RUN THE QSP CIRCUITS'''
    Ep_PLOT(Plist, Qlist, E0, n, a, b, data,  ax=axes, )
    W_PLOT(philist, n,  data, conv=defconv, ax=axes, )
    #HAAHR_PLOT(philistr, n,  data, conv='R', ax=axes, )

    return 

def PLOT_QSPSIM_WITH_DATA(Plist, Qlist, E0, data, a, b, ifsave=False, withcomp=False):
    fig, axes = plt.subplots(1, figsize=(12, 8))
    
    Ep_PLOT(Plist, Qlist, E0, n, a, b, data,  ax=axes, )
    if withcomp==True:
        axes.scatter(data, np.real(lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+ 1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))))
        axes.scatter(data,np.imag(lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+ 1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))))

    if ifsave==True:
        plt.savefig(save_path+"REQSP_HS_for_t="+str(t)+".png")
    else:
        plt.show()
    return


#PLOT_QETSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=False, )


#PLOT_REALQSPSIM_WITH_DATA(Plist, Qlist, E0, n, inst_tol, data)

# def GQSP_SIM(thetalist, philist, lambd, U, convent=np.array([[1], [1]])/2):
#     clength=len(thetalist)
#     Ul=len(U)
#     SystIdent=np.identity(Ul)

#     CzU=np.kron(np.array([[1, 0], [0, 0]]), U)+np.kron(np.array([[0, 0], [0, 1]]), SystIdent)
#     Uqsp=np.kron(RU(thetalist[0], philist[0], lambd), SystIdent)
    
#     for i in range(1, clength):
#         iterU=np.kron(RU(thetalist[i], philist[i]), SystIdent)@CzU
#         Uqsp=iterU@Uqsp

    
#     projtoconvent=np.kron(convent@np.conj(convent).T, SystIdent)@Uqsp@np.kron(convent@np.conj(convent).T, SystIdent)
#     Ured=np.trace(projtoconvent.reshape(2, Ul, 2, Ul), axis1=0, axis2=2)   
#     return Ured



# Pcoeffs, Qcoeffs, Podd, Qeven, Gn=REINDEX(a, b, d, c)

# P_approx=lpf.LAUR_POLY_BUILD(Pcoeffs, np.int32(Gn/2), np.exp(1j*data)**2)+np.exp(1j*data)*lpf.LAUR_POLY_BUILD(Podd, np.int32(Gn/2), np.exp(1j*data)**2)#*np.exp(1j*thdata)**n
# Q_approx=np.exp(1j*data)*lpf.LAUR_POLY_BUILD(Qcoeffs, np.int32(Gn/2), np.exp(1j*data)**2) #*np.exp(1j*data)**n
# P_approx=(np.exp(1j*data)**n)*np.exp(1j*data)*lpf.LAUR_POLY_BUILD(Pcoeffs, np.int32((n-1)/2), np.exp(1j*data)**2)#*np.exp(1j*thdata)**n
# Q_approx=np.exp(1j*data)*Q_POLY_BUILD(Qcoeffs, n, np.exp(1j*data)**2) #*np.exp(1j*data)**n

# targetfcn=(np.cos(xdata*t)/2*np.exp(1j*data)+1j*np.sin(xdata*t)/2)*np.exp(1j*data)**n

#targP=np.exp(1j*data)*np.cos(xdata*t)/2#*np.exp(1j*thdata)**n
#targQ=1j*np.sin(xdata*t)/2 #*np.exp(1j*data)**n

# print(P_approx*np.conj(P_approx)+Q_approx*np.conj(Q_approx))
# Pgenlist=Pcoeffs+Podd
# Qgenlist=Qcoeffs+Qeven
# print(len(Pgenlist))
# thetalist,philist, lambd=COMPUTE_R(Pgenlist, Qgenlist, Gn)

# GU=GQSP_SIM(thetalist, philist, lambd, U, convent=np.array([[1], [1]])/np.sqrt(2))

# targetfcn=np.exp(1j*xdata*t)/2*np.exp(1j*data)**2
# axes[3].plot(data, np.real(P_approx), marker='.', label=r'$\mathcal{P}_{real}$')  
# axes[3].plot(data, np.real(targP),label=r'$e^{it\lambda}_{real}$', )
# axes[3].plot(data, np.imag(targQ), marker='.', label=r'$e^{it\lambda}_{imag}$',)  
# axes[3].plot(data, np.imag(Q_approx),label=r'$\mathcal{Q}_{imag}$' )
# axes[0].legend()


# GQSP=np.array(GQSPl)

#axes[3].scatter(Hevals, np.imag(GQSP),color='seagreen', marker='1',label=r'$\bra{\lambda}U_{gqsp}\ket{\lambda}_{imag}$')
#axes[3].scatter(Hevals, np.real(GQSP), color='gold',marker='1',label=r'$\bra{\lambda}U_{gqsp}\ket{\lambda}_{real}$')
# axes[3].set_title('Compare QSP result to the origional Function'+r"$|\mathcal{A}-\bra{\lambda}U_{\Phi}\ket{\lambda}|_{\max}=$"+ str((mf.NORM_CHECK(QETA, FCNA), 2)))


