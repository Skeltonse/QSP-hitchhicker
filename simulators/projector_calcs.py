#!/usr/bin/env python
# coding: utf-8

# Given a set of degree $n$ Laurent polynomials $a, b, c, d$, computes a projector set $P_{2n}$

import numpy as np
import functions.laur_poly_fcns as lpf 
import matplotlib.pyplot as plt
from scipy.fft import fftn
import simulators.matrix_fcns as mf

def BUILD_C(a, b, c, d, n, dofourier=False):
    """
    Defines matrix valued coefficients indexed [2, 2, 2*n+1]. c, d can be switched

    inputs:
    a, b, c, d: 2n+1 length np arrays storing coefficient lists of real-on-circle Laurent polynomials
    n: float, max degree of a, b, c, d
    dofourier: option to Fourier transform the coefficient lists, not currently implemented/tested

    return: 2x2x2n+1 length np array, stores the coefficent list of $F^{2n}(z)$ used in the QSP decomposition step
    """

    C=np.zeros([2, 2, 2*n+1], dtype=complex)
    C[0, 0, :]=a+d*1j
    C[1, 1, :]=a-d*1j
    C[0, 1, :]=c+b*1j
    C[1, 0, :]=-c+b*1j
    
    ##PAD THE INITIAL FUNCTIONS TO RE-INDEX F(z)->F(z^2)
    Ci=np.zeros([2, 2, 4*n+1], dtype=complex)
    for i in range(0, 2*n+1):
        Ci[:, :, 2*i]=C[:, :, i]

    return Ci

def BUILD_PQCi(C, m):
    """
    Computes the mth projectors P, Q, and returns then along with the matrix valued coefficient list for (m-1)

    inputs:
    C: 2x2x2m+1 np array, storing all the coefficients of 'intermediate' matrix-valued function $F^{m}(z)$ in the decomposition step
    m: float, degree

    returns:
    P, Q: 2x2 matrices with matrices close to a projector
    Cn: 2x2x2m-1 np array, storing all the coefficients of '$F^{m-1}(z)$ 
    """
    Pun=C[:, :, 2*m].T.conj()@C[:, :, 2*m]
    P=Pun/np.trace(Pun)
    Q=np.identity(2)-P
    
    Cn=np.zeros([2, 2, 2*m-1], dtype=complex)
    for i in range(0, 2*m-1):
        Cn[:, :, i]=C[:, :, i]@Q+C[:, :, i+2]@P
    return P, Q, Cn


def PROJECTOR_CHECK(M, epsi):
    """
    Checks that a matrix in M(2) is epsilon-close to a projector.
    returns warning if not

    inputs:
    M: 2x2 numpy array
    epsi: float, precision of answer

    returns:
    P: 2x2 array, the projector M is 'closest' too
    """
    evals, evecs=np.linalg.eig(M)
    cprob=np.where(np.imag(evals)>epsi)[0]
    sol=np.where(np.real(evals)>epsi)[0]
    
    import_vec=evecs[:,sol[0]].reshape(2, 1)
    P=import_vec@np.conj(import_vec).T
    
    return P

    
def PROJECTIFY_PQCi(P, C, epsi, m,):
    """
    Finds a projector epsilon-close to a given matrix.

    inputs:
    P 2x2 np array, a matrix which is known to be close to a projctor
    C: 2x2x 2m+1 np array, coefficient list of 'intermediate' matrix-valued function $F^{m}(z)$ in the decomposition step
    epsi: float, the required precision of the solution
    m: float, the degree of the Laurent polynomial that C builds
    
    """
    
    P1=PROJECTOR_CHECK(P, epsi)
    Q1=np.identity(2)-P1
    
    Cn=np.zeros([2, 2, 2*m-1], dtype=complex)
    for i in range(0, 2*m-1):
        Cn[:, :, i]=C[:, :, i]@Q1+C[:, :, i+2]@P1

    return P1, Q1, Cn

def UNIFY_PLIST(a, b, c, d, n, epsi):
    """
    computes the projector sets P, Q from input real-on-circle coefficient lists a, b, c,d.
    Ensures that projectors are exactly unitary
    
    inputs:
    a, b, c, d: length 2n+1 np arrays, coefficient lists of each Laurent polynomial
    n: float, max Laurent polynomial degree
    
    Returns:
    P, Q: 2x2x2n length np arrays storing 2n projectors
    E0: 2x2 np array, unitary matrix 
    """
    Plist=np.zeros([2, 2, 2*n], dtype=complex)
    Qlist=np.zeros([2, 2, 2*n], dtype=complex)
    
    Ci=BUILD_C(a, b, c, d, n)
    
    for l in range(0,2*n):
        m=2*n-l
        P, Q, Cinext=BUILD_PQCi(Ci, m)
        P1, Q1, Ci1=PROJECTIFY_PQCi(P,Ci, epsi,m)
        
        Plist[:, :,m-1]=P1
        Qlist[:, :, m-1]=Q1
        Ci=Ci1

    E0=Ci1[:, :, 0]
    return Plist, Qlist, E0

def BUILD_PLIST(a, b, c, d, n, dofourier=False):
    """
    computes the projector sets P, Q from input real-on-circle coefficient lists a, b, c,d.

    inputs:
    a, b, c, d: length 2n+1 np arrays, coefficient lists of each Laurent polynomial
    n: float, max Laurent polynomial degree
    
    Returns:
    P, Q: 2x2x2n length np arrays storing 2n (approximate) projectors
    E0: 2x2 np array, unitary matrix 
    """
    Plist=np.zeros([2, 2, 2*n], dtype=complex)
    Qlist=np.zeros([2, 2, 2*n], dtype=complex)
    
    Ci=BUILD_C(a, b, c, d, n, dofourier)
    for l in range(0,2*n):
        m=2*n-l
        P, Q, Ci=BUILD_PQCi(Ci, m)
        Plist[:, :,m-1]=P
        Qlist[:, :, m-1]=Q
        
    E0=Ci[:, :, 0]
    return Plist, Qlist, E0

def Ep_CALL(t, Plist, Qlist, E0, n):
    """
    returns the product of the sequence of the E_p sequence defined by Plist, E0 for some t.
    Recall that this is an estimate of the QSP sequence affect of some matrix with eigenvaue 2t

    inputs:
    t: scalar or array, points to evaluate the QSP sequence
    P, Q: 2x2x2n length np arrays storing 2n projectors
    E0: 2x2 np array, unitary matrix

    return:
    float or array, the a(2t)+ib(2t) part of E_0\prod E_p(t)
    """
    E=E0[:, :]
    for i in range(0, 2*n):
        Ei=(t*Plist[:, :, i] + 1/t*Qlist[:, :, i])
        E=E@Ei

    conv=np.array([[1], [1]])/np.sqrt(2)
    val=conv.T@E@conv
    return val[0, 0]

def Ep_PLOT(Plist, Qlist, E0, n, czlist, szlist, theta, ax=None, just_vals=False, **plt_kwargs):
    """
    Plots the E_p and Laurent polynomial expressions for the same function

    inputs:
    P, Q: 2x2x2n length np arrays storing 2n projectors
    E0: 2x2 np array, unitary matrix
    n : degree of Laurent polynomial
    szlist, czlist: 2n+1 length np arrays, the odd and even coefficient lists for the Laurent polynomial f
    theta : np array with points in varible \theta to compute E_p plot on
    ax : TYPE, optional
        DESCRIPTION. The default is None.


    """
    Eplist=np.zeros(len(theta),dtype=complex)
    
    for t, th in enumerate(theta):
        Eplist[t]=Ep_CALL(np.exp(1j*th/2), Plist, Qlist, E0, n)
    fl=1j*lpf.LAUR_POLY_BUILD(szlist, n, np.exp(1j*theta))+lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*theta))

    if just_vals==True:
        return Eplist
    else:
        ax.plot(theta, np.real(fl),label=r'$\mathcal{A}_{Re}(\theta)$', marker='.', **plt_kwargs)
        ax.plot(theta, np.real(Eplist),  linewidth=1, label=r'$E_p(\theta/2)_{Re}$', **plt_kwargs)
        ax.plot(theta, np.imag(fl),label=r'$\mathcal{A}_{Im}(\theta)$',  marker='.',**plt_kwargs)
        ax.plot(theta, np.imag(Eplist),  linewidth=1, label=r'$E_p(\theta/2)_{Im}$', **plt_kwargs)
        ax.legend()
        ax.set_title(r'plots for $f(\theta)$, $E_p(e^{i\theta/2})$')
    return Eplist

def SU2_CHECK(a, b, c, d, n, test=np.exp(1j*np.pi/6)):
    """
    check if a, b, c, d builds an element of SU(2)

    inputs:
    a, b, c, d: length 2n+1 np arrays, the coefficient lists
    n: float, the max degree of a, b, c, d
    test: a point in U(1) to test
    """
    
    ta=lpf.LAUR_POLY_BUILD(a, n, test)
    tb=lpf.LAUR_POLY_BUILD(b, n, test)
    tc=lpf.LAUR_POLY_BUILD(c, n, test)
    td=lpf.LAUR_POLY_BUILD(d, n, test)
    tF=np.array([[ta+td*1j,tc+tb*1j ],[-tc+tb*1j, ta-td*1j]])
    return (tF@np.conj(tF).T)
