"""
Functions to process, compute, or check Laurent polynomial lists
Also includes functions useful for plotting data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time 
import tikzplotlib

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')


'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import solvers.Wilson_method as wm
from simulators.projector_calcs import Ep_PLOT, SU2_CHECK, UNIFY_PLIST, BUILD_PLIST
import simulators.matrix_fcns as mf


###FUNCTIONS TO CHECK POLYNOMIAL CONDITIONS OR PROCESS COEFFICIENTS
def ab_PROCESS(a, b, n,  theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16), plots=True, ax=None, **plt_kwargs):
    """
    builds the coefficient array for F(z)=1-a^2(z)-b^2(z) from coeff arrays a, b real valued Laurent polys.
    Checks if F is real-on-circle within tolerance and prints warning if not
    (optional) plots F 
    
    inputs:
    a, b: 2n+1 length np arrays storing coefficient lists
    n: float, degree of polynomial
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    plots: True/False determines whether to generate a plot
    ax, **plt_kwargs: optional argements for plotting

    returns:
    a, b: same as inputs
    calcFc: length 4n+1 np array with coefficients of the Fejer input polynomial $1-a^2-b^2$
    calF: calF: np array with values of the Fejer input polynomial for each value in \theta
    """
    ###GET RID OF ANY INCONSEQUENTIALLY SMALL COEFFICIENTS###
    coeffcutoff=10**(-16)
    while abs(a[0])<coeffcutoff and abs(b[0])<coeffcutoff:
        a=a[1:2*n]
        b=b[1:2*n]
        n=n-1
        
    ###BUILD LIST FOR THE FEJER INPUT POLYNOMIAL, $1-a^2-b^2$###
    cz2list=lpf.LAUR_POLY_MULT(a, a)
    sz2list=lpf.LAUR_POLY_MULT(b, b)
    add1=np.append(np.append(np.zeros(2*n), 1), np.zeros(2*n))
    abunc=cz2list+sz2list
    abun=lpf.LAUR_POLY_BUILD(abunc, 2*n,  np.exp(1j*theta))
    calFc=add1-abunc
    
    ###CHECK THAT THE ANSWER IS REAL###
    calF=lpf.REAL_CHECK(calFc, 2*n, theta=theta, tol=tol)

    if plots==True:
        if ax is None:
            ax = plt.gca()
        ax.plot(theta, np.real(calF),label=r"$1-a^2(z)-b^2(z)$", linestyle="-", **plt_kwargs)
        ax.plot(theta, np.real(lpf.LAUR_POLY_BUILD(a, n,  np.exp(1j*theta))),label=r"$a(z)$", **plt_kwargs)
        ax.plot(theta, np.real(lpf.LAUR_POLY_BUILD(b, n,  np.exp(1j*theta))),label=r"$b(z)$", **plt_kwargs)
        ax.set_title(r'Plots for Fejer Prob Input Poly')
        ax.set_xlabel(r'$\theta$')

    
    return a, b, calFc, calF, n

def cd_PROCESS(gamma, a, b, n, theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16), plots=True, ax=None, **plt_kwargs):
    """
    builds the coefficient lists for c, d

    inputs:
    gamma: 2n+1 length np array, the coefficients of the solution to the fejer problem
    a, b: 2n+1 length np arrays storing coefficient lists
    n: float, degree of polynomial
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    plots: True/False determines whether to generate a plot
    ax, **plt_kwargs: optional argements for plotting

    returns:
    c, d: 2n+1 coefficient lists representing the real and imaginary parts of the Fejer solution
    probflag: binary variable, 0 is default and 1 signals that there is a problem with the solution
    """
    probflag=0

    ###GET c, d AS REAL-ON-CIRCLE POLYNOMIALS AND CHECK PROPERTIES###
    c=(gamma+np.flip(gamma))/2
    d=-1j*(gamma-np.flip(gamma))/2 
    lpf.REAL_CHECK(c, n, theta, tol, 'c')
    lpf.REAL_CHECK(d, n,  theta, tol, 'd')

    ###CHECK THE SUM OF SQUARED a, b, c, d IS ALWAYS 1###
    Fcheckc=lpf.LAUR_POLY_MULT(a, a)+lpf.LAUR_POLY_MULT(b, b)+lpf.LAUR_POLY_MULT(c, c)+lpf.LAUR_POLY_MULT(d, d)
    Fcheck=lpf.LAUR_POLY_BUILD(Fcheckc, 2*n,  np.exp(1j*theta))
    if plots==True:
        if ax is None:
            ax = plt.gca()
        ax.plot(theta, np.real(Fcheck), label=r'$a^2(z)+b^2(z)+c^2(z)+d^2(z)$')
        ax.plot(theta, np.real(lpf.LAUR_POLY_BUILD(c, n, np.exp(1j*theta))), label=r'$\gamma_{re}(z)$')
        ax.plot(theta, np.real(lpf.LAUR_POLY_BUILD(d, n, np.exp(1j*theta))), label=r'$\gamma_{imag}(z)$')
        ax.set_xlabel(r"$\theta$")
        
    else:
        problems=np.where(abs(Fcheck-1)>tol)
        if problems[0]!=[]:
            print('probelm, a, b, c, do not obey constraint')
            print(problems)
            probflag=1
    return c, d, probflag


def F_CHECK(calFc, n, theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16),  rtn='none', fcnname='F'):
    """
    checks if Laurent poly with coefficients calFc is real and positive on the unit circle

    inputs:
    calFc: length 2n+1 coefficient list
    n: float, degree of the Laurent polynomial
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    rtn: option to return the array of the polynomial evalauted along points in \theta
    fcnname: string labelling the polynomial being evaluated    
    """
    calF=lpf.LAUR_POLY_BUILD(calFc, n, np.exp(theta*1j))
    if any(abs(np.imag(calF))>tol):
        print(r'Warning'+fcnname+ 'has imaginary terms')
        print("largest term norm is", max(abs(np.imag(calF))))
    elif any(np.real(calF)<=0):
        print(r'Warning, '+ fcnname + ' has negative real terms')
        print("largest negative term norm is", min((np.real(calF))))
    if rtn=='fcn_vals':
        return calF
    else:
        return

def GAMMA_PROCESSING(gamma, n, calF, theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16),plots=False, ax=None, **plt_kwargs):
    """
    Computes a normalization factor for the Fejer solution. This should be unnecessary for the Wilson method
    
    inputs:
    gamma: 2n+1 length np array, the coefficients of the solution to the fejer problem
    n: float, degree of polynomial
    calF: 4n+1 length np array, the coefficients of Fejer input 
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    plots: True/False determines whether to generate a plot
    ax, **plt_kwargs: optional argements for plotting

    returns:
    the average subnormalization needed to make the solution work 
    """
    coeffp=lpf.LAUR_POLY_MULT(gamma, np.flip(gamma))
    ###generates values over $\theta\in[-\pi, \pi]$ and checks for problems
    calFp=F_check(coeffp, 2*n, tol, theta, rtn='fcn_vals')
    if ax is None:
        ax = plt.gca()

    alpha_list=np.real((calF/calFp))  
    
    if plots==True:
        if ax is None:
            ax = plt.gca()
        ax.plot(theta, np.real(calFp), label=r'$\gamma(z)\gamma(1/z)$', **plt_kwargs)
        ax.plot(theta, np.real(calFp)*np.mean(alpha_list), label='normalized solution', **plt_kwargs)
        ax.set_title(r"Compare Wilson solution guess to $1-|\mathcal{A}|^2$")
    return np.mean(alpha_list)
              
def GAMMA_CHECK(gamma,  n, calF, theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16), show='fcns', ax=None, **plt_kwargs):
    """
    Checks that the Fejer solution is real-on-circle and positive, prints warnings if not
    
    inputs:
    gamma: 2n+1 length np array, the coefficients of the solution to the fejer problem
    n: float, degree of polynomial
    calF: 4n+1 length np array, the coefficients of Fejer input 
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    show: string determining which plot to show.
    --'fcns' displays the Fejer input against solution
    --'diff' displays the difference between at each point
    --all other strings will skip plots
    ax, **plt_kwargs: optional argements for plotting

    """
    coefft=lpf.LAUR_POLY_MULT(gamma, np.flip(gamma))
    
    calFt=lpf.LAUR_POLY_BUILD(coefft, 2*n, np.exp(theta*1j))


    if any(abs(np.imag(calFt))>tol):
        print(r'Warning, $\mathcal{F}$ has imaginary terms')
        print("largest term norm is", max(abs(np.imag(calFt))))
    elif any(np.real(calFt)<=0):
        print(r'Warning, $\mathcal{F}$ has negative real terms')
        print("largest negative term norm is", min(np.real(calFt)))
        
    if show=='fcns':
        if ax is None:
            ax = plt.gca()
        ax.plot(theta, np.real(calFt), label=r'$\gamma(z)\gamma(1/z)$ ', **plt_kwargs)
        
        ax.set_title(r'Compare $\gamma$ to $\mathcal{F}$')
    elif show=='diff':
        if ax is None:
            ax = plt.gca()
        ax.plot( np.real(calF)-np.real(calFt), **plt_kwargs)
        ax.set_title(r'Difference between $\gamma$ and $\mathcal{F}')
    return
    

'''BEGIN FINDING THE SOLUTION: '''
def PARAMETER_FIND(czlist, szlist,n, data,epsi=10**(-14), defconv='ad', ifdecomp=True, tDict={}, plots=False):
    """
    Runs the Fejer solver for each instance. checks incoming polynomial lists, builds Feer input $\mathcal{F}(z)$,
    solves for solution, check sit,  computes $c(z), d(z)$ and checks them.
    computes projectors defining the QSP circuit
    (optional) displays plots for each step and/or times the completion step 

    inputs:
    czlist, szlist: 2n+1 length np arrays storing coefficient lists
    n: float, degree of polynomial
    data: np array with a list of points
    epsi: tolerance for error
    defconv: determines whether d is defined with the reciprocal or the anti-reciprocal part
    ifdecomp: True/False determines whether to complete the decomposition step
    tdict: defines a dictionary to store the solution
    plots: True/False determines whether to generate a plot
    axeschoice, **plt_kwargs: optional argements for plotting
    tdict:
    
    returns:
    a, b: same as inputs
    calcFc: length 4n+1 np array with coefficients of the Fejer input polynomial $1-a^2-b^2$
    calF: calF: np array with values of the Fejer input polynomial for each value in \theta
    """
    ###BOUNDED ERROR ONLY: DEFIN THE PRECISION OF THE FEJER STEP###
    epsifejer=epsi

    if plots==True:
        fig, axeschoice = plt.subplots(2, figsize=(12, 6))
    else:
        axeschoice=[None, None]

    ###CHECK UNPUT POLYNOMIALS AND SOLVE THE COMPLETION STEP###
    a, b, calFc, calF, n=ab_PROCESS(czlist, szlist, n,  theta=data, tol=epsi, plots=plots,  ax=axeschoice[0])
    t0=time.perf_counter()
    gammaW, itW, initgamma, nu, Ttilde=wm.WILSON_LOOP_WCHECK(np.real(calFc), 2*n, nu=epsifejer, init="Wilson", datatype='float')
    t1=time.perf_counter()
    c, d, probflag=cd_PROCESS(gammaW, a, b,n, tol=epsi,plots=plots, ax=axeschoice[1])
    
    if plots==True:
        GAMMA_CHECK(gammaW,  n, calF, theta=data, tol=10**(-14), show='fcns', ax=axeschoice[0])
    else:
        GAMMA_CHECK(gammaW,  n, calF, theta=data, tol=10**(-14), show='none',ax=axeschoice[0])
       
    if plots==True:
        axeschoice[0].legend()
        axeschoice[1].legend()
        plt.show()
    solDict={'soltime': t1-t0, 'solit':itW,'degree':n ,'a':a, 'b':b,  'c': c, 'd': d, 'gamma': gammaW, 'initialguess':initgamma, 'wilsontol': nu, 'rerunflag':probflag, "Ttilde": Ttilde}
    tDict.update(solDict)
    
    if ifdecomp==False:
        return tDict
        
    '''THIS IS WHERE THE CONVENTION KICKS IN'''
    if defconv=='ad':
        Plist, Qlist, E0=UNIFY_PLIST(a, b, d, c, n, 64*epsifejer)
    elif defconv=='ac':
        Plist, Qlist, E0=UNIFY_PLIST(a, b, c, d, n,64*epsifejer)
    
    tDict['Plist']=Plist
    tDict['Qlist']=Qlist
    tDict['E0']=E0
    
    return Plist, Qlist, E0, a, b, c, d, n, tDict

def NORM_EXTRACT(a, b, c, d, n, data, epsi):
    """
    Computes the difference between the QSP simulation calculation and the polynomial a+ib at each point
    beginning from coefficient lists, computes the projector set, the QSP solution for each point,
    and the difference between them
    inputs:
    a, b, c d length 2n+1 numpy arrays, should be real-on-circle, pure Laurent polynomials
    n: float, max degree
    data: the number of \theta points used
    epsi: the solution tolerance for computing P, Q, E0

    return: np array with the Euclidean trace distance at each point
    """
    ftestreal=lpf.LAUR_POLY_BUILD(a, n, np.exp(1j*data))
    ftestimag=lpf.LAUR_POLY_BUILD(b, n, np.exp(1j*data))

    Plist, Qlist, E0=BUILD_PLIST(a, b, c, d, n)
    Wlist=Ep_PLOT(Plist, Qlist, E0,n, a, b, data, just_vals=True)
    
    return mf.NORM_CHECK(Wlist, ftestreal+1j*ftestimag)

def NORM_EXTRACT_FROMP(Plist, Qlist, E0,a, b, n,  fcnvals, data):
    """
    Computes the difference between the QSP simulation calculation and the polynomial a+ib at each point
    given known projector sets computes the QSP solution for each point
    
    inputs:
    Plist, Qlist: np 2x2x2n array, must have PList[2, 2, j] be a projector for every j (and for Qlist)
    E0: np 2x2 array, must be in SU(2)
    a, b, length 2n+1 numpy arrays, should be real-on-circle, pure Laurent polynomials
    n: float, max degree
    data: the number of \theta points used

    return: np array with the Euclidean trace distance at each point
    """
    Wlist=Ep_PLOT(Plist, Qlist, E0,n, a, b, data, just_vals=True)
    return  mf.NORM_CHECK(Wlist, fcnvals)


def LS_FIT(x, a, b):
    '''
    simple linear function for least squares fit. In LAtEX: $y=ax+b$
    inputs + output are all floats
    '''
    y = a*x +b
    return y


def LS2_FIT(x, a, b, c):
    '''
    simple polynomial function for least squares fit. In LaTeX: $y=bx^a+c$
    inputs + output are all floats
    '''
    y=b*x**a+c
    return y

