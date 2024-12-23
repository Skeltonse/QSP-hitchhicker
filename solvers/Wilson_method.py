#!/usr/bin/env python
# coding: utf-8

# The Wilson method for solving a Fejer problem. Assumes the function is normalized

# In[ ]:

import numpy as np
import time 
import scipy.signal as sig

def WILSON_GUESS_ADJ(n):
    """
    The guess suggested in Wilson's paper, with the addition of a high degree term - sets the initial guess \gamma_0 to a high degree function fcn with a constant term.
    0<const2<const1 guarentees that this doesn't have roots on the unit circle
    For simplicity, assumes coefficients are up to O(n) and positive

    input: float n, degree of initial guess

    return:
    length n+1 np array, the coefficient list of the initial guess
    """
    zc=np.zeros(n)
    constc=np.random.randint(low=1, high=n, size=1)
    gam0=np.append(constc, zc)
    const2=0.1*np.random.randint(low=1, high=constc, size=1)
    gam0[n]=const2
    
    return gam0 

def WILSON_GUESS(n, coeff):
    """
    The guess suggested in Wilson's paper, with the addition of a high degree term - sets the initial guess \gamma_0 to a constant fcn with \gamma_{0, 0}>0.
    Assumes coefficient is up to O(n) and positive

    input: float n, degree of initial guess

    return:
    length n+1 np array, the coefficient list of the initial guess
    """
    zc=np.zeros(n)
    # constc=np.random.randint(low=1, high=n, size=1)
    delta=0.01
    B=(3-2*delta)*(n+1)/(1-delta)
    gammaguessabove=B/(n+1)**2
    
    constc=np.random.randint(low=gammaguessabove, high=n**3*gammaguessabove, size=1)
    gam0=np.append(constc, zc)
    return gam0  

def T_c_COMP(gam, n, datatype):
    """Computes $T_1, T_2, c for a particular $\gamma_i$.

    inputs:
    gam: length n+1 np array, coefficient list $(g_0,...g_n)$
    n: float, degree of polynomial
    
    returns:
    T: np aray for matrix $T_1+T_2$
    c: np array for vector $c=T_1\gamma$
    """
    T1=np.zeros([n+1, n+1], dtype=datatype)
    T2=np.zeros([n+1, n+1], dtype=datatype)

    ###create arrays storing each i index value
    i_ind=np.arange(0, n+1, 1, dtype=int)

    for j_ind, gamj in enumerate(gam):
        plusind=i_ind[j_ind+i_ind<=n].astype(int)
        minusind=i_ind[(j_ind-i_ind<=n) & (0<=j_ind-i_ind)].astype(int)
        T1[plusind, j_ind]=gam[j_ind+plusind]
        T2[minusind, j_ind]=gam[j_ind-minusind]
    T=T1+T2
    c=T1@gam.T
    return T, c

def POLY_MULT(coeffa, coeffb, datatype=float):
    """
    produces the coefficient list of the product of polynomials a, b from their coefficient lists.
    Assumes a, b are polynomials of different degrees.

    inputs:
    coeffa, coeffb: np arrays storing coefficients of each polynomial
    """

    return sig.convolve(coeffa, coeffb, method='fft')

def POLY_BUILD(coeff, n, z):
    """
    computes the float-point value of a polynomial from its coefficient list.
    

    input:
    coeff: np array storing coefficients of the polynomial
    n: float, degree of the polynomial
    z: float or np array, the point the polynomial is evaluated at

    return:
    polyval: float or array, the evaluated polynomial
    """
    polyval=0
    for l in range(0,n+1):
        polyval=polyval+coeff[l]*z**l
    return polyval


def l_inf_NORM_DIFF(coefflist1, coefflist2, n, z=np.exp(1j*np.linspace(-np.pi,np.pi, 100))):
    
    """
    computes the l-\infty norm of the difference between functions in C^2 beginning from coefficient lists

    inputs:
    coeff1, coeff2: n+1 length np arrays, coefficient lists of polynomials
    n: degree of polynomial

    return: the maximum distance between the pointwise functional values
    """
    vallist1=POLY_BUILD(coefflist1, n, z)
    vallist2=POLY_BUILD(coefflist2, n, z)
    obj=vallist1-vallist2
    normlist=np.sqrt(np.real(obj)**2+np.imag(obj)**2)
    return np.max(normlist)

def WILSON_LOOP_WDATA(coeff, n, nu=10**(-16), init="Wilson", method='linsolve', datatype=float):
    """
    RUNS THE WILSON SOLVER TO SOLVE A FEJER PROBLEM. 

    inputs:
    coeff: 2n+1 length np array, the coefficent list of the Laurent polynomial to solve
    n: float, the degree of the Laurent polynomial
    nu: float, the solver tolerance
    init: string determining which initial guess is used.
    --"Wilson" uses the default guess in WILSON_GUESS, a constant polynomial
    --"Wilson2" uses WILSON_GUESS_ADJ, a high degree polynomial with constant term
    --any other string uses GUESS to compute an initial guess
    method: string determining which linear solver is used
    --"linsolve" uses numpy's default linear solver
    --"gauss-seidel" calls that method
    --any other strin calls the conjugate gradient method
    datatype: needs to match the datatype of coefff

    output: the gamma solution and number of iterations to the solution
    """
    ###SET THE MAXIMUM NUMBER OF ITERATIONS###
    itW=0
    i_max=100#np.int64(n)
    
    ###GENERATE AN INITIAL GUESS AND COMPUTE T, c###
    if init=="Wilson":
        gam0=WILSON_GUESS(n, coeff)
    elif init=="Wilson2":
        gam0=WILSON_GUESS_ADJ(n)
    else:
        gam0=GUESS(n)
    initgam=gam0
    T, c=T_c_COMP(gam0, n, datatype)
    
    ###TRUNCATE THE COEFFICIENT LIST TO NON-NEGATIVE INDICES###
    a=coeff[n:]
    
    ##MAIN LOOP
    for i in range(1, i_max+1):
        if method=='linsolve':
            ngam=np.linalg.solve(T, c+a)
            #delt=np.linalg.solve(T, a)
            #ngam=delt+gam0/2
            
        elif method=='gauss_seidel':
            ngam=gauss_seidel(T, c+a)
            
        else:
            ngam=conjugate_gradient(T, c+a)

        ###COULD USE A SOMETIMES STRICTER CRITERIA OF SUCCESS: if abs(ngam-gam0).all()<nu:
        if l_inf_NORM_DIFF(ngam, gam0, n)<nu:
            itW=i
            print('Wilson solution found after ' + str(i) + ' iterations')
            break
        elif i==i_max:
            print('Wilson solution not found after ' + str(i) + ' iterations')
        else:
            gam0=ngam
            T, c=T_c_COMP(gam0, n, datatype)
            
        if i%100==0:
            print('wilson step', i)
            
    return ngam,  itW, initgam, nu

def WILSON_LOOP_WCHECK(coeff, n, nu=10**(-14), init="Wilson", method='linsolve', datatype=float):
    """
    RUNS THE WILSON SOLVER TO SOLVE A FEJER PROBLEM. 

    inputs:
    coeff: 2n+1 length np array, the coefficent list of the Laurent polynomial to solve
    n: float, the degree of the Laurent polynomial
    nu: float, the solver tolerance
    init: string determining which initial guess is used.
    --"Wilson" uses the default guess in WILSON_GUESS, a constant polynomial
    --"Wilson2" uses WILSON_GUESS_ADJ, a high degree polynomial with constant term
    --any other string uses GUESS to compute an initial guess
    method: string determining which linear solver is used
    --"linsolve" uses numpy's default linear solver
    --"gauss-seidel" calls that method
    --any other strin calls the conjugate gradient method
    datatype: needs to match the datatype of coefff

    output: the gamma solution and number of iterations to the solution
    """
    ###SET THE MAXIMUM NUMBER OF ITERATIONS###
    itW=0
    i_max=100#np.int64(n)
    
    ###GENERATE AN INITIAL GUESS AND COMPUTE T, c###
    if init=="Wilson":
        gam0=WILSON_GUESS(n, coeff)
    elif init=="Wilson2":
        gam0=WILSON_GUESS_ADJ(n)
    
    initgam=gam0
    T, c=T_c_COMP(gam0, n, datatype)
    
    ###TRUNCATE THE COEFFICIENT LIST TO NON-NEGATIVE INDICES###
    a=coeff[n:]
    
    ##MAIN LOOP
    for i in range(1, i_max+1):
        if method=='linsolve':
            ngam=np.linalg.solve(T, c+a)
            
        elif method=='gauss_seidel':
            ngam=gauss_seidel(T, c+a)
            
        else:
            ngam=conjugate_gradient(T, c+a)

        ###COULD USE A SOMETIMES STRICTER CRITERIA OF SUCCESS: if abs(ngam-gam0).all()<nu:
        if l_inf_NORM_DIFF(ngam, gam0, n)<nu:
            itW=i
            # print('Wilson solution found after ' + str(i) + ' iterations')
            break
        elif i==i_max:
            print('Wilson solution not found after ' + str(i) + ' iterations')
        else:
            gam0=ngam
            T, c=T_c_COMP(gam0, n, datatype)
            
        if i%100==0:
            print('wilson step', i)
    Ttilde, ctilde=T_c_COMP(ngam, n, datatype)    
    return ngam,  itW, initgam, nu, Ttilde

###STACK EXCHANGE SOURCED LINEAR SOLVERS, SOMETIMES HAVE AN ADVANTAGE OVER np default solver###
def jacobi(A, b, tolerance=1e-12, max_iterations=10000):
   ###code from Texas PGE 
    x = np.zeros_like(b, dtype=np.double)
    
    T = A - np.diag(np.diagonal(A))
    
    for k in range(max_iterations):
        
        x_old  = x.copy()
        
        x[:] = (b - np.dot(T, x)) / np.diagonal(A)
        
        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break
            
    return x

def gauss_seidel(A, b, tolerance=1e-12, max_iterations=10000):
    
    x = np.zeros_like(b, dtype=np.double)
    
    #Iterate
    for k in range(max_iterations):
        
        x_old  = x.copy()
        
        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]
            
        #Stop condition 
        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break
            
    return x

def conjugate_gradient(A, b):
    r = b 
    k = 0
    x = np.zeros(A.shape[-1])
    while np.linalg.norm(r) > 1e-12 :
        if k == 0:
            p = r
        else: 
            gamma = - (p @ A @ r)/(p @ A @ p)
            p = r + gamma * p
        alpha = (p @ r) / (p @ A @ p)
        x = x + alpha * p
        r = r - alpha * (A @ p)
        k =+ 1
    return x

