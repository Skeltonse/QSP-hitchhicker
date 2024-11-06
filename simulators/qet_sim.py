# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:42:41 2023

@author: Shawn Skelton
QET simulator for Unitary U and projector set P
"""

import numpy as np

def VFINDER(P, Q):
    """
    Computes the jth Unitray set V, V^{\dag} corresponding to
    the P_jth projector for Haah QSP
    V: \ket{0}\rightarrow \ket{p}
    Well tested (|p_0|^2+|p_1|^2=1, \theta\in\mathbb{R}, th_{ac}+th_{ca}=0, ra<1, correct sign)

    inputs:
    P, Q : [2, 2] numpy arrays storing the projectors for Complex QSP
    tol: float, tolerance of the 

    Returns
    V,Vd : [2, 2] numpy arrays; unitaries to act on the ancillary qubit for QET

    """
    ###GET RADI###
    ra=np.sqrt(P[0, 0])
    rb=np.sqrt(Q[0, 0])
    rc=np.sqrt(P[1, 1])
    rd=np.sqrt(Q[1,1])

    ###GET RELATIVE ANGLES: there is a degree of freedom so we can always set \theta_a=\theta_b=0###
    thca=1j*np.log(P[0, 1]/ra/rc)
    thdb=1j*np.log(Q[0, 1]/rb/rd)

    tha=0
    thb=0
    thc=thca+tha
    thd=thdb+thb
    
    V=np.array([[ra*np.exp(1j*tha), rb*np.exp(1j*thb)], [rc*np.exp(1j*thc), rd*np.exp(1j*thd)]])
    Vd=np.array([[ra*np.exp(-1j*tha), rc*np.exp(-1j*thc)], [rb*np.exp(-1j*thb), rd*np.exp(-1j*thd)]])
    return V, Vd

def COMPLEX_QET_SIM(U, Plist, Qlist, E0, convent=np.array([[1], [1]])/np.sqrt(2), ):
    """
    The main function for computing QET circuits as in https://quantum-journal.org/papers/q-2019-10-07-190/
    Builds each CU gate wrt projectors. Usuing standard E_p def from the reference, I use the following controlled opers
    -for odd indexed projectors (indexed even in code), the controlled operation is $C_pU=P\otimes I + Q\otimes U$
    -for even indexed projectors (odd in code), the controlled operation is $C_pU^{\dag}=P\otimes U + Q\otimes I$
    

    inputs:
    U : np array, the unitary function whose eigenvalues we will transform
    Plist, QList : projector sets stored as [2, 2, 2*n+1] numpy arrays
    E0: 2x2 np array, a unitary on the ancillary
    convent : vactor as np array,  the ancillary basis element we want to measure to obtain the correct function
    --default is np.array([[1], [1]])/np.sqrt(2).

    Returns
    -------
    The simulated QSP circuit, post-selected on the desired ancillary mmnt

    """
    ###DEFINE THE OPERATOR FOR THE CUMULATIVE UNITARY EVOLUTION ON THE CIRCUIT###
    Ul=len(U)
    SystIdent= np.identity(Ul)
    E=np.kron(E0, SystIdent)
    
    ##EXTRACT THE NUMBER OF PROJECTORS P: there will be 2n P's indexed 0...2n-1, so we always get an even sequence###
    p0,p1,p2=Plist.shape

    ###MAIN LOOP###
    for ind in range(int(p2/2)):
        # E=E@(np.kron(Plist[:, :, 2*ind], SystIdent)+np.kron(Qlist[:, :, 2*ind],U))@(np.kron(Qlist[:, :, 2*ind+1], SystIdent)+np.kron(Plist[:, :, 2*ind+1],np.conj(U).T))
        E=E@(np.kron(Qlist[:, :, 2*ind], SystIdent)+np.kron(Plist[:, :, 2*ind],U))@(np.kron(Plist[:, :, 2*ind+1], SystIdent)+np.kron(Qlist[:, :, 2*ind+1],np.conj(U).T))


    ###TRACE OUT THE ANCILLARY SYSTEM###
    projtoconvent=np.kron(convent@np.conj(convent).T, SystIdent)@E@np.kron(convent@np.conj(convent).T, SystIdent)
    Ured=np.trace(projtoconvent.reshape(2, Ul, 2, Ul), axis1=0, axis2=2)
    
    return Ured
    
def COMPLEX_QET_SIM2(U, Plist, Qlist, E0, convent=np.array([[1], [1]], dtype=complex)/np.sqrt(2)):
    """
    Computing QET circuits as in https://quantum-journal.org/papers/q-2019-10-07-190/
    Builds each C_1U gate and then intersperses unitary rotations
    -for odd indexed steps (indexed even in code), the controlled operation is $C_pU=P\otimes I + Q\otimes U=VC_0U{\dag}V^{\dag}$.
    --instead we define $V\ket{0}\rightarrow\ket{q}, V\ket{1}\rightarrow\ket{p}$ so that $VC_1U{\dag}V^{\dag}=C_pU$
    -for even indexed steps (odd in code), the controlled operation is $C_pU^{\dag}=P\otimes U + Q\otimes I=VC_1U{\dag}V^{\dag}$    

    inputs:
    U : np array, the unitary function whose eigenvalues we will transform
    Plist, QList : projector sets stored as [2, 2, 2*n+1] numpy arrays
    E0: 2x2 np array, a unitary on the ancillary
    convent : vactor as np array,  the ancillary basis element we want to measure to obtain the correct function
    --default is np.array([[1], [1]])/np.sqrt(2).

    Returns
    -------
    The result of the simulated QSP circuit (no post-selection
    
    """
    Ul=len(U)
    SystIdent= np.identity(Ul)
    E=np.kron(E0, SystIdent)
    
    ###DEFINE CONTROLLED U, U^{\dag} AS USUAL WITH RESPECT TO THE COMPUTATIONAL BASIS 
    CtrlU=np.kron(np.array([[1, 0], [0, 0]]), SystIdent)+np.kron(np.array([[0, 0], [0, 1]]),U)
    CtrlUd=np.kron(np.array([[1, 0], [0, 0]]), SystIdent)+np.kron(np.array([[0, 0], [0, 1]]),np.conj(U).T)

    ###DEFINE THE OPERATOR FOR THE CUMULATIVE UNITARY EVOLUTION ON THE CIRCUIT###
    E=np.kron(E0, SystIdent)
    
    ##EXTRACT THE NUMBER OF PROJECTORS P: there will be 2n P's indexed 0...2n-1, so we always get an even sequence###
    p0,p1,p2=Plist.shape
    
    ##MAIN LOOP: 
    for ind in range(int(p2/2)):
        V, Vd=VFINDER(Plist[:, :, 2*ind], Qlist[:, :, 2*ind])
        V2,Vd2=VFINDER(Qlist[:, :, 2*ind+1], Plist[:, :, 2*ind+1])
        
        E=E@np.kron(V, SystIdent)@CtrlU@np.kron(Vd@V2, SystIdent)@CtrlUd@np.kron(Vd2, SystIdent) 
    
    return E

def QET_MMNT(UPhi, evec):
    """
    Basically just a simple measurement

    input:
    Uphi: nxn np array, the unitary evolution of the system
    eveec: vector as 1xn numpy array, the state we measure wrt to

    returns: float, value of the measurement
    """
    mmnt=np.conj(evec).T@(UPhi)@evec
    return mmnt
