import numpy as np
from simulators.angle_calcs import *
import simulators.matrix_fcns as mf


def GCNOT(Pi):
    """
    Generalized Controlled NOT gate, where the control is on projector Pi, rather than on 1
    from definition 2 of Gilyen
    """
    g=len(Pi)
    return np.kron(mf.sigmaX, Pi)+np.kron(mf.I, np.identity(g)-Pi)

def CPIROT(CNOTPi, a, phi):
    """
    Controlled Rotation about I-2Pi subspace, computed as in F1.b from Gilyen (hence the -ve sign on phi)
    CNOTPi: np array, a controlled not operation usually computed with GCNOT
    a: rank of Pi
    phi: angle in [-pi, pi] (or [0, 2pi], it won't matter)
    """
    return CNOTPi@np.kron(PRz(-phi), np.identity(a))@CNOTPi

def PROJ_TO_SYST(UBE, Pi, Pit,syst_dim, a):
    """
    projects to the Pi, \Tilde{\Pi} subspace and then returns the reduced density matrix over the system qubits
    UBE: unitary block encoding as np array with dimensions syst_dim+a
    Pi, PIt: projectors as np array with dimensions a
    a, syst_dim : the dimenions of the system and projectors (so qubits are log2(a), log2(syst_dim))
    """
    projtosyst=np.kron(Pit, np.identity(int(syst_dim)))@UBE@np.kron(Pi, np.identity(int(syst_dim)))
    densitymatrix=np.trace(projtosyst.reshape(a, int(syst_dim),a, int(syst_dim)), axis1=0, axis2=2)
    
    return densitymatrix 

def QSVT_SIM(UBE, Pi, Pit, philist, n, cvec=np.array([[1], [1]]/np.sqrt(2))):
    """
    QSVt simulation. UBE is assumed to be a block encoding, so Pi, Pit have the same dimension.
    UBE, Pi, PIt: all numpy arrays. UBE must be unitary with dimensions (systm_dim +a) and Pi, Pit should be projectors in dimension 2a.
    """
    ###Get dimensions###
    a=max(np.shape(Pi)[0], 2)
    nBE=np.shape(UBE)[0]
    syst_dim=nBE-a
    

    ###Get U and CNOT wrt projectors in the larger space, 1st qubit is QSVT ancilla###
    ULspace=np.kron(mf.I, UBE)
    UdLspace=np.kron(mf.I, np.conj(UBE).T)

    #Builds controlled operations in $\mathbb{C}_2\mathbb{C}_m and acts trivially in \mathbb{C}_n,
    CNOTPi=GCNOT(Pi)
    CNOTPit=GCNOT(Pit)
    SYSTid=np.identity(syst_dim)
    
    ###THE QSVT CIRCUIT### 
    ##eventually this will be the result
    Uphi= np.identity(2*nBE)
    
    ##odd QSVT circuit
    if n%2==1:
        print('odd')
         ##(the indices look wrong: they're not, but I had to swap parity bc python starts at 0 and Thrm 17 starts at 1)
        for ind, phi in enumerate(philist):
            if ind%2==0:
                Uphi=Uphi@np.kron(CPIROT(CNOTPit, a, phi),SYSTid)@ULspace
            elif ind%2==1:
                UPhi=Uphi@np.kron(CPIROT(CNOTPi, a,phi), SYSTid)@UdLspace
            
        #print(np.kron(I, Pit)@Uphi@np.kron(I, Pi))
        #  densitymatr=Pit@np.trace(Uphi.reshape(2, UBl, 2, UBl),axis1=0, axis2=2)@Pi
        projtoBE=np.kron(cvec@cvec.T, np.identity(nBE))@Uphi@np.kron(cvec@cvec.T, np.identity(nBE))
        UPhiBE=np.trace(projtoBE.reshape(2, nBE, 2, nBE),axis1=0, axis2=2)
        syst_density_matrix=PROJ_TO_SYST(UPhiBE, Pi, Pit,syst_dim, a)
    
    
    ##even QSVT circuit
    elif n%2==0: 
        for ind, phi in enumerate(philist):
            if ind%2==0:
                Uphi=Uphi@np.kron(CPIROT(CNOTPi, a, phi), SYSTid)@UdLspace
            elif ind%2==1:
                Uphi=Uphi@np.kron(CPIROT(CNOTPit, a, phi), SYSTid)@ULspace
        projtoBE=np.kron(cvec@np.conj(cvec).T, np.identity(nBE))@Uphi@np.kron(cvec@cvec.T, np.identity(nBE))
        UPhiBE=np.trace(projtoBE.reshape(2, nBE, 2, nBE),axis1=0, axis2=2)
        syst_density_matrix=PROJ_TO_SYST(UPhiBE, Pi, Pi,syst_dim, a)
        
        
    return syst_density_matrix

def QSVT_CALL(x,philist, n):
    UBE=mf.U_BUILD1(np.array([x, np.sqrt(1-x**2)]), np.array([[1, 0], [0, 1]]))
    
    UPhi=QSVT_SIM(UBE, np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]]), philist, n, cvec=np.array([[1], [0]]))

    return np.real(UPhi[0, 0])

def QSVT_PLOT_COMP_HAAH(philist, n,  x,  ax=None,  **plt_kwargs):
    """
    Plots the QSVT calculation for an array of x values embedded in....

    Parameters
    ----------
    philist : The philist for Laurent polynomial $f$.
    
    n : degree of Laurent polynomial
    x : data array, usually from [0, 1]
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    Wlist=np.zeros(len(x),dtype=complex)
        
    for xind, xval in enumerate(x):
        Wlist[xind]=QSVT_CALL(np.cos(np.arccos(xval/2)), philist, 2*n,) 
    if ax is None:
        ax = plt.gca()
    ax.plot(x, np.real(Wlist),label=r'$U_{QSVT, Re}$', **plt_kwargs)
    # ax.plot(x, np.imag(Wlist),label='Wimag', **plt_kwargs)
    ax.legend()

def QSVT_PLOT(philist, n,  x,  ax=None,  **plt_kwargs):
    """
    Plots the QSVT calculation for an array of x values embedded in....

    Parameters
    ----------
    philist : The philist for Laurent polynomial $f$.
    
    n : degree of Laurent polynomial
    x : data array, usually from [0, 1]
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    Wlist=np.zeros(len(x),dtype=complex)
        
    for xind, xval in enumerate(x):
        Wlist[xind]=QSVT_CALL(xval, philist, n,) 
    if ax is None:
        ax = plt.gca()
    ax.plot(x, np.real(Wlist),label=r'$U_{QSVT, Re}$', **plt_kwargs)
    # ax.plot(x, np.imag(Wlist),label='Wimag', **plt_kwargs)
    ax.legend()
   

