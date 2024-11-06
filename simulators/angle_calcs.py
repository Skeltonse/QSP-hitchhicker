# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:41:35 2023

@author: Shawn Skelton

Write a function to obtain the angle set from a projector set P. 
Should assume we know which two of a, b, c, d are recip.
"""
import numpy as np
import matplotlib.pyplot as plt
import simulators.matrix_fcns as mf

def U1_LOC(phic, phis):
    "takes two expressions for an angle phi, from arccos and arcsin resp. returns \phi\in[0, 2\pi]"
    ###begin assuming phi\in quad I, II so phic is accurate
    phi=phic
    ###rotate through quads, changing angle value as neccessary
    if -np.pi/2<phis<0:
        phi=2*np.pi-phic
    return phi
    
def PROJ_TO_ANGLE(Plist, E0, n, recip='ad', tol=10**(-16)):
    'converts Haah projector set Plist into Gilyén angle set philist. Report values \phi\in[0, 2\pi]'
    philist=np.zeros(2*n, dtype=complex)
    phi0=0
    'W_x convention, E_0 should be related to Pauli Z'
    
    if recip=='ad':
        if any(PAULI_CHECK(E0, "Z", tol))!=1:
            print("warning, E0 has the wrong form to be a Pauli Z rotation")
            print(PAULI_CHECK(E0, "Z", tol))
            print(E0)
         
        "compute phi_0, to make the summation conveniant at the end, introduce a negative sign (which cancels with the one from log)"
        phi0=-(1j*np.log(E0[0, 0]))
        # print(E0-PRz(phi0))
        "now compute the angles for j=1...2n"
        for j in range(0, 2*n):
            philist[j]=1j/2*np.log(2*Plist[1, 0, j])
            # print(Plist[:,:,j]-PRz(philist[j])@np.array([[1, 1], [1, 1]])@PRz(-philist[j])/2)
        phi0p=phi0+philist[0]
        intermed=np.diff(philist)
        philast=-philist[-1]
        
#     elif recip=='ab':
#         "W_z convention, E_0 should be related to Pauli X Rotation"
#         if abs(E0[0, 0])>1+tol or abs(E0[1, 1])>1+tol:
#             print('warning, E0 has incorrect form')
#             print(E0)
#         phi0c=np.arccos(E0[0, 0])
#         phi0s=np.arcsin(-1j*E0[0, 1])
#         phi0=U1_LOC(phi0c, phi0s)-
#         for j in range(0, 2*n):
#             phic=np.arccos(np.sqrt(Plist[0, 0, j]))
#             phis=np.arcsin(np.sqrt(Plist[1, 1, j]))
#             philist[j]=U1_LOC(phic, phis)
    elif recip=='ac':
        "W_y convention, E_0 should be related to Pauli Y Rotation"
        if any(PAULI_CHECK(E0, "Y", tol))!=1:
            print("warning, E0 has the wrong form to be a Pauli Y rotation")
            
        "get the first angle"
        phi0c=np.arccos(E0[0, 0])
        phi0s=np.arcsin(E0[0, 1])
        "to make the summation conveniant at the end, introduce a negative sign here"
        phi0=-U1_LOC(phi0c, phi0s)
        
        "now compute the angles for j=1...2n"
        for j in range(0, 2*n):
            phic=np.arccos(np.sqrt(Plist[0, 0, j]))
            phis=np.arcsin(np.sqrt(Plist[1, 1, j]))
            philist[j]=U1_LOC(phic, phis)
#             print(SENSIBLE_MATRIX(Plist[:,:,j]-PRy(philist[j])@np.array([[1, 0], [0, 0]])@PRy(-philist[j])))
    "Take the difference of {\phi_0,\phi_1...,\phi_n}, and append \phi_n to it."
    phiplist=np.append(np.append(phi0p, intermed), philast)
    
    return np.real(phiplist)

###define all the Pauli rotations
def PRx(phi):
    return np.array([[np.cos(phi), 1j*np.sin(phi)], [1j*np.sin(phi), np.cos(phi)]])

def PRz(phi):
    return np.array([[np.cos(phi)+1j*np.sin(phi), 0], [0, np.cos(phi)-1j*np.sin(phi)]])

def PRy(phi):
    return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

def W_CALL(x, philist, n, recip='ad'):
    E=np.identity(2)
    conv=np.array([[1], [1]])/np.sqrt(2)
    if recip=='ad':
        #PRxVAL=PRx(theta)
        PRxVAL=np.array([[x, 1j*np.sqrt(1-x**2)], [1j*np.sqrt(1-x**2), x]])
        E=E@PRz(philist[0])
        for i in range(1, n+1):
            E=E@PRxVAL@PRz(philist[i])
            
        # conv=np.array([[1], [1]])/np.sqrt(2)
    
#     elif recip=='ab':
#         E=E0
#         for i in range(0, 2*n):
#             Ei=PRx(theta)@PRz(philist[i])
#             #print(SENSIBLE_MATRIX(Ei@np.conj(Ei).T))
#             E=E@Ei
#         conv=np.array([[1], [0]])
#     elif recip=='ac':
#         PRzval=PRz(theta)
#         E=E@PRy(philist[0])
#         for i in range(1, 2*n+1):
#             E=E@PRzval@PRy(philist[i])
#         conv=np.array([[1], [1]])/np.sqrt(2)
# #     print(SENSIBLE_MATRIX(E))
    
    elif recip=='R':
        PRrval=np.array([[x, np.sqrt(1-x**2)], [np.sqrt(1-x**2), -x]])
        # E=E@PRz(philist[0])
        for i in range(0, n):
            E=E@PRz(philist[i])@PRrval
        conv=np.array([[1], [0]])
    
    val=conv.T@E@conv
    
    return val[0, 0]

def PAULI_CHECK(E0, conv, tol=10**(-16)):
    E0=mf.SENSIBLE_MATRIX(E0, tol)
    iftrue=np.array([0, 0, 0])
    if conv=='Z':
        iftrue[0]=abs(E0[0, 0]*E0[1, 1]-1)<tol
        iftrue[1]=abs(E0[0, 1])<tol
        iftrue[2]=abs(E0[1, 0])<tol
    if conv=='Y':
        if E0[np.where(np.imag(E0)>tol)].size>0:
            print("warning, these rows have imaginary values", E0[np.where(np.imag(E0)>tol)])
        iftrue[0]=abs(E0[0, 0]**2+E0[0, 1]**2-1)<tol
        iftrue[1]=abs(E0[0, 0]-E0[1,1])<tol
        iftrue[2]=abs(E0[0, 1]+E0[1, 0])<tol
    return iftrue

def Ep_CALL(z, Plist, Qlist, E0, n):
    """Builds the \Prod E_P(z) series which defines Haah's QSP circuit. Should not be actually used outside of testing Projector sets or simple conparisons"""
    E=E0[:, :]
    for i in range(0, 2*n):
        Ei=(z*Plist[:, :, i] + 1/z*Qlist[:, :, i])
        
        E=E@Ei
#     print(SENSIBLE_MATRIX(E))
    conv=np.array([[1], [1]])/np.sqrt(2)
    val=conv.T@E@conv
    return val[0, 0]

def Wx_TO_R(philistx, tol=10**(-14)):
#     d=len(philist)
#     philistr=np.zeros([d])
#     philistr[0]=philist[0]+(2*d-1)*np.pi/4
#     philistr[-1]=philist[-1]-np.pi/4
#     philistr[1:d-1]=philist[1:d-1]-np.pi/2
    d=len(philistx)-1
    philistr=np.zeros([d])
    if any(abs(np.imag(philistx))>tol):
        print('warning, philist has nontrivial imaginary components')
    philistr[0]=np.real(philistx[0]+philistx[d]+np.pi/2*(d-1))
    philistr[1:]=np.real(philistx[1:d]-np.pi/2)
    return philistr


# def SENSIBLE_MATRIX(A, tol=10**(-16)):
#     """function to make reading matrix results easier. checked against a simple example"""
#     Ar=np.where(abs(np.real(A))>tol, np.real(A), 0)
#     Ai=np.where(abs(np.imag(A))>tol, np.imag(A), 0)
#     return Ar+Ai*1j


def W_PLOT(philist, n,  x, conv='R', ax=None,  **plt_kwargs):
    """
    Plots the E_p and Laurent polynomial expressions for the same function

    Parameters
    ----------
    philist : The philist for Laurent polynomial $f$.
    
    n : degree of Laurent polynomial
    theta : data array, usually from [-\pi, \pi]
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    Wlist=np.zeros(len(x),dtype=complex)
        
    for xind, xval in enumerate(x):
        Wlist[xind]=W_CALL(xval, philist, n, conv) 
    if ax is None:
        ax = plt.gca()
    ax.plot(x, np.real(Wlist),label='Wre', **plt_kwargs)
    ax.plot(x, np.imag(Wlist),label='Wimag', **plt_kwargs)
    ax.legend()
    
def HAAHW_PLOT(philist, n,  x,  conv='R', ax=None,  **plt_kwargs):
    """
    Plots the E_p and Laurent polynomial expressions for the same function

    Parameters
    ----------
    philist : The philist for Laurent polynomial $f$.
    
    n : degree of Laurent polynomial
    theta : data array, usually from [-\pi, \pi]
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    Wlist=np.zeros(len(x),dtype=complex)
        
    for xind, xval in enumerate(x):
        Wlist[xind]=W_CALL(np.cos(np.arccos(xval/2)), philist, 2*n, conv) 
    if ax is None:
        ax = plt.gca()
    ax.plot(x, np.real(Wlist),label='Wre', **plt_kwargs)
    ax.plot(x, np.imag(Wlist),label='Wimag', **plt_kwargs)
    ax.legend()

def HAAHR_PLOT(philist, n,  theta, conv='R', ax=None, just_vals=False,  **plt_kwargs):
    """
    Plots the E_p and Laurent polynomial expressions for the same function

    Parameters
    ----------
    philist : The philist for Laurent polynomial $f$.
    
    n : degree of Laurent polynomial
    theta : data array, usually from [-\pi, \pi]
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    Wlist=np.zeros(len(theta),dtype=complex)
        
    for thetaind, thetaval in enumerate(theta):
        Wlist[thetaind]=W_CALL(np.cos(thetaval/2), philist, 2*n, conv) 
    if ax is None:
        ax = plt.gca()
    if just_vals==True:
        return np.real(Wlist)
        
    ax.plot(theta, np.real(Wlist),label=r'$QSP_{R, re}(x)$', **plt_kwargs)
    # ax.plot(x, np.imag(Wlist),label='Wimag', **plt_kwargs)
    ax.legend()
