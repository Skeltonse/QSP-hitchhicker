"""
Useful functions on matrix valued objects
"""
import numpy as np
from scipy.stats import unitary_group

sigmaX=np.array([[0,1],[1, 0]])
sigmaZ=np.array([[1,0],[0, -1]])
sigmaY=np.array([[0,-1j],[1j, 0]])
I=np.identity(2)

def SENSIBLE_MATRIX(A, tol=10**(-16)):
    """
    function to make reading matrix results easier, sets any very small matrix elements to zero

    inputs:
    A: n x n complex np. array
    tol: tolerance of the solution
    """
    Ar=np.where(abs(np.real(A))>tol, np.real(A), 0)
    Ai=np.where(abs(np.imag(A))>tol, np.imag(A), 0)
    return Ar+Ai*1j


def random_hermitian_matrix(N, ρ):
    """
    Return a random real matrix with spectral radius ≈ ρ,
    taken from https://gist.github.com/goerz/cd369b9d02a8c1dbd0b2a95bd9fd5545.
    N: float, dimension of square matrix
    p: float, spectral redius

    returns: a Hermitian matrix
    """
    σ = 1/np.sqrt(N)
    X = np.random.normal(0.0, scale=σ, size=(N, N))
    X_dag = X.conjugate().transpose()
    H = 0.5*(X + X_dag) / np.sqrt(2)
    return ρ * H

def NORM_CHECK(qeta, fnca):
    """
    computes the distance between points in C^2 and returns the maximum

    inputs:
    qeta, fnca: np arrays, assumed to be lists of functional values at different points in the complex plane

    return: the maximum distance between the pointwise functional values
    """
    obj=qeta-fnca
    normlist=np.sqrt(np.real(obj)**2+np.imag(obj)**2)
    return np.max(normlist)

def UNITARY_BUILD(H, return_evals=False):
    """
    Builds e^{iarccos(H)}, a suitable oracle for complex QSP

    input:
    H: Hermitian matrix
    return_evals: option to return the eigenvalues of H and the eigenvectors

    return:
    U: np array, the unitary QSP oracle for H
    Uevals: np array with the eigenvalues of U
    """
    dims=np.shape(H)[0]
    Hevals, evecs=np.linalg.eig(H)
    Uevals=np.exp(1j*np.arccos(Hevals))
    
    U=np.zeros([dims, dims], dtype=complex)
    for t in range(0, dims):
      U=U+Uevals[t]*evecs[:, t][:, np.newaxis]@np.conj(evecs[:, t][:, np.newaxis]).T
      
    if return_evals==True:
        return U, Uevals, evecs, Hevals
    
    return U, Uevals

def OPNORM(U):
    """
    Returns the operator norm of unitary U 

    input:
    U : Unitary matrix as np array

    output: operator norm
    """
    opnormU=np.sqrt(max(abs(np.linalg.eigvals(np.conj(U).T@U))))
    return opnormU

