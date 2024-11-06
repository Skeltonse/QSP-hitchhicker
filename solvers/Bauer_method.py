#!/usr/bin/env python
# coding: utf-8

# Bauer Method for solving the Fejer Problem: much less testing than the Wilson method, so use with caution

import numpy as np


def npr_CHOL(npr, coeff, n, tol=10**(-16)):
    """linalg.cholesky is lower triangular by default...but a negligibly small complex numbers are a pain with linalg.cholesky
    checked the indexing of eigs in the documentation. See """
    T=TOEP_BUILD(npr, coeff, n)
    ###get the eigenvalues and eigenvectors of T, and then process###
    lambs, vecs=np.linalg.eig(T)
    
    ##get rid of neglibible complex factors
    if np.any(np.imag(lambs)>tol):
        print('warning, this matrix has nontrivial imaginary eigs')
    else:
        lambs=np.real(lambs) 
    ##get rid of negligible real factors
    if np.any((lambs)<-tol):
        print('warning, this matrix has nontrivial negative eigs')
        print(np.real(lambs))
    else:
        lambs=np.where(abs((lambs))>tol, lambs,0).astype('float')
    #lambs=np.where(abs(np.imag(lambs))>tol, lambs, np.real(lambs).astype('float'))
    
    ##assemble sqrt(D), V##
    sqrt_Lamb=np.zeros([npr+1, npr+1])
    V=np.zeros([npr+1, npr+1], dtype=complex)
#     Lamb=np.zeros([npr+1, npr+1]) #checking
    
    for i in range(0, len(lambs)):
        sqrt_Lamb[i, i]=np.sqrt(lambs[i])
        V[:, i]=vecs[:, i]
       # Lamb[i, i]=(lambs[i])#checking
    #print(SENSIBLE_MATRIX(T-V@sqrt_Lamb@sqrt_Lamb@V.conj().T))  #checking
    
    ##compute the QR decomposition##
    sqrLambVdag=sqrt_Lamb@V.conj().T
    Q, LT=np.linalg.qr(sqrLambVdag)
    
    return LT.T


    
def TOEP_BUILD(npr, coeff, n):
    """Creates an (n'+1, n'+1) matrix $T_{n'}$ indexed $i, j=0...n'$.
    Inputs are $n', (a_{-n},....a_{n}), n$
    This takes negligible time even for $n, n'>>1$
    tested with simple functions, explicitly checked row/column assignments"""
    T=np.zeros([npr+1, npr+1], dtype=complex)
    for i in range(0, npr+1):
        for j in range(0, npr+1):
            if i-j<-n:
                break
            elif i-j>n:
                continue
            else:
                T[i, j]=coeff[i-j+n]
    return T
def BAUER_LOOP(coeff, n, nu):
    #set the maximum number of iterations
    i_max=n**4
    itB=0
    #get the first Cholesky decomp and its last row
    L0=npr_CHOL(n+1, coeff, n)
    coeff0=L0[-1, 1:]
    Llen=n+2
    
    for i in range(2, i_max+1):
        coeffi=npr_CHOL(n+i, coeff, n)[-1, i:]
        Li=npr_CHOL(n+i, coeff, n)
        
        if max(abs(coeffi-coeff0))<nu:
            print('Bauer solution found after ' + str(i) + ' iterations')
            Llen=i+n
            itB=i
            break
        elif i==i_max:
            print('Bauer solution not found after ' + str(i) + ' iterations')
        else:
            coeff0=coeffi
            #coeffi=Li[-1, i:] #indexing note: the (j+1)st coefficent is in index j
    
    gamma=np.zeros(n+1, dtype=complex)
    ##grab the coefficients of $\gamma$ from the 0th column of L_{m+i+1}
    for r in range(0, n+1):
        gamma[r]=Li[r,0]
    return gamma, itB

