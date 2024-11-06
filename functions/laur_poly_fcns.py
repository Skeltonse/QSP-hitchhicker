#!/usr/bin/env python
# coding: utf-8

# Functions to check Laurent polynomials. All coefficient lists are numpy arrays ordered $\left(c_{-n}, c_{-n+1},...c_0, ...c_n\right)$. Default tolerance is $10^{-16}$ unless otherwise specified
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def REAL_CHECK(coeff, n, theta=np.linspace(-np.pi,np.pi, 100),  tol=10**(-16), fcnname='function', giveprecision=False):
    """
    CHECKS IF A LAURENT POLY IS REAL-ON-CIRLCE.
    Computes values on the unit circle, for Laurent polynomial with coefficients a of degree n.
    returns an error if any are larger than the set tolerance

    inputs:
    coeff: length 2n+1 np array, coefficient list of a Laurent polynomial
    n: float, degree of the Laurent polynomial
    theta: np array of points to check functional values
    tol: float, tolerance of solution
    fcnname: string naming the function being checked
    giveprecision: True/False option to return the max error in the Laurent polynomial

    return:
    coeffQ: np array of function values, Laurent polynomial evaluated at each point in theta
    
    """
    coeffQ=LAUR_POLY_BUILD(coeff, n,  np.exp(1j*theta))
    
    if max(abs(np.imag(coeffQ)))>tol:
        print('warning, '+ fcnname +' has nontrivial imaginary component')
        print(max(abs(np.imag(coeffQ))))
        if giveprecision==True:
            return coeffQ, max(abs(np.imag(coeffQ)))
    else:
        if giveprecision==True:
            return coeffQ, tol
    return coeffQ

# def ab_PROCESS(a, b, n, show=True,  theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16), ax=None, **plt_kwargs):
#     """
#     builds the coefficient array for F(z)=1-a^2(z)-b^2(z) from coeff arrays a, b real valued Laurent polys.
#     Checks if F is real-on-circle and computes Fmax from rescaled \mathcal{A}
#     """
#     cz2list=LAUR_POLY_MULT(a, a)
#     sz2list=LAUR_POLY_MULT(b, b)
#     add1=np.append(np.append(np.zeros(2*n), 1), np.zeros(2*n))

#     ###Need to have $a^2+b^2<1$ over the unit circle
#     abunc=cz2list+sz2list
#     abun=LAUR_POLY_BUILD(abunc, 2*n,  np.exp(1j*theta))
    
#     calFc=add1-abunc
    
#     calF=REAL_CHECK(calFc, 2*n, theta=theta, tol=tol)

#     if show==True:
#         if ax is None:
#             ax = plt.gca()
#         ax.plot(theta, np.real(calF),label=r"$1-a^2(z)-b^2(z)$", **plt_kwargs)
#         # ax.set_title(r'Plots for Fejer Functions')
        
#         # ax.text(theta[0], 0.8, r'$F_{max}=$'+str(np.round(np.sqrt(maxF), 2)))

    
#     return a, b, calFc, calF

# def cd_PROCESS(gamma, n, a, b, tol=10**(-16), theta=np.linspace(-np.pi,np.pi, 100), plots=True, ax=None, **plt_kwargs):
#     probflag=0
#     # print('n',n)
#     # print('length gamma', len(gamma))
#     c=(gamma+np.flip(gamma))/2
#     d=-1j*(gamma-np.flip(gamma))/2
    
#     REAL_CHECK(c, n, tol, theta, 'c')
#     REAL_CHECK(d, n, tol, theta, 'd')
#     Fcheckc=LAUR_POLY_MULT(a, a)+LAUR_POLY_MULT(b, b)+LAUR_POLY_MULT(c, c)+LAUR_POLY_MULT(d, d)
#     REAL_CHECK(Fcheckc, 2*n, tol, theta, fcnname='F')
#     Fcheck=LAUR_POLY_BUILD(Fcheckc, 2*n,  np.exp(1j*theta))
#     if plots==True:
#         if ax is None:
#             ax = plt.gca()
#         ax.plot(theta, np.real(Fcheck), label=r'$a^2(z)+b^2(z)+c^2(z)+d^2(z)$')
#         ax.plot(theta, np.real(LAUR_POLY_BUILD(c, n, np.exp(1j*theta))), label='c')
#         ax.plot(theta, np.real(LAUR_POLY_BUILD(d, n, np.exp(1j*theta))), label='d')
#         ax.legend()
#         # print('largest to smallest c val', min(LAUR_POLY_BUILD(c, n, np.exp(1j*theta))), max(LAUR_POLY_BUILD(c, n, np.exp(1j*theta))))
#         # print('largest to smallest d val', min(LAUR_POLY_BUILD(d, n, np.exp(1j*theta))), max(LAUR_POLY_BUILD(d, n, np.exp(1j*theta))))
#         #ax.set_title('Check if Fejer Problem is solved')
#     else:
#         problems=np.where(abs(Fcheck-1)>tol)
#         if problems[0]!=[]:
#             print('probelm, c, d pair does not obey constraint')
#             print(problems)
#             probflag=1
#     return c, d, probflag
# # print(np.array([0, 1, 2, 3, 4, 5]))
# # cd_PROCESS(np.array([0, 1, 2, 3, 4, 5]), 5, np.zeros(6), np.zeros(6))

# def F_check(calFc, n, tol=10**(-16),  theta=np.linspace(-np.pi,np.pi, 100), rtn='none'):
#     '''checks if Laurent poly with coefficients calFc is real and positive on the unit circle'''
#     calF=LAUR_POLY_BUILD(calFc, n, np.exp(theta*1j))
#     if any(abs(np.imag(calF))>tol):
#         print(r'Warning, $\mathcal{F}$ has imaginary terms')
#         print(max(abs(np.imag(calF))))
#     elif any(np.real(calF)<=0):
#         print(r'Warning, $\mathcal{F}$ has negative real terms')
#     if rtn=='fcn_vals':
#         return calF
#     else:
#         return

# def GAMMA_PROCESSING(gamma, n, calF, tol=10**(-16),theta=np.linspace(-np.pi,np.pi, 100), plots=False, ax=None, **plt_kwargs):
#     coeffp=LAUR_POLY_MULT(gamma, np.flip(gamma))
#     ###generates values over $\theta\in[-\pi, \pi]$ and checks for problems
#     calFp=F_check(coeffp, 2*n, tol, theta, rtn='fcn_vals')
#     if ax is None:
#         ax = plt.gca()

#     alpha_list=np.real((calF/calFp))  
    
#     if plots==True:
#         #ax.plot(theta, np.real(calF), label='og function to solve', **plt_kwargs)
#         ax.plot(theta, np.real(calFp), label=r'$\gamma(z)\gamma(1/z)$', **plt_kwargs)
#         ax.plot(theta, np.real(calFp)*np.mean(alpha_list), label='normalized solution', **plt_kwargs)
#         ax.legend()
#         # ax.set_ylim(-1.2, 1.2)
#         ax.set_title(r"Compare Wilson solution guess to $1-|\mathcal{A}|^2$")
#     #return np.real(calFp)*np.mean(alpha_list), np.mean(alpha_list)
#     return np.mean(alpha_list)
              
# def GAMMA_CHECK(gamma, alpha,calF, n, show='fcns', tol=10**(-16), theta=np.linspace(-np.pi,np.pi, 100), ax=None, **plt_kwargs):
#     coefft=LAUR_POLY_MULT(gamma, np.flip(gamma))
#     coeffnormed=LAUR_POLY_MULT(alpha*gamma, np.flip(alpha*gamma))
    
#     calFt=LAUR_POLY_BUILD(coefft, 2*n, np.exp(theta*1j))
#     calFnormed=LAUR_POLY_BUILD(coeffnormed, 2*n, np.exp(theta*1j))


#     if any(abs(np.imag(calFt))>tol):
#         print(r'Warning, $\mathcal{F}$ has imaginary terms')
#         print(max(abs(np.imag(calFt))))
#     elif any(np.real(calFt)<=0):
#         print(r'Warning, $\mathcal{F}$ has negative real terms')
#         print(min(np.imag(calFt)))
#     if show=='fcns':
#         ax.plot(theta, np.real(calF), label=r'$\mathcal{F}(z)$', **plt_kwargs)
#         ax.plot(theta, np.real(calFt), label=r'$\gamma(z)\gamma(1/z)$ ', **plt_kwargs)
#         ax.plot(theta, np.real(calFnormed), label=r'$\alpha\gamma(z)\gamma(1/z)$ ', **plt_kwargs)
#         ax.legend()
#         ax.set_title(r'Compare $\gamma$ (w/o) normalization to $\mathcal{F}')
#     elif show=='diff':
#         ax.plot( np.real(calF)-np.real(calFt), **plt_kwargs)
#         ax.plot( np.real(calF)-np.real(calFnormed), **plt_kwargs)
#         ax.set_title(r'Difference between $\gamma$ (w/o) normalization and $\mathcal{F}')
#     return
    

# Functions to build Laurent polynomials
def CHEBY_POLY_BUILD(coeff, n, th, term='c'):
    """
    computes the float-point value of a Fourier expansion from its Chebyshev coefficient list.

    inputs:
    coef: length n+1 coefficient array
    n: float, degree of polynomial
    theta: float or np array of points in \theta\in (-\pi, \pi) to check functional values
    term: string, determines whether the cosinal or sinusoidal term is being computed
    --'s': computes the sin expansion
    --'c' computes the cosine expansion
    """
    polyval=0
    if term=='c':
        for l in range(0, n+1):
            polyval=polyval+coeff[l]*np.cos(l*th)
    elif term=='s':
        for l in range(0, n+1):
            polyval=polyval+coeff[l]*np.sin(l*th)
    return polyval

def GET_LAURENT(clist, slist, n):
    """
    Converts the coefficient list of polynomials in x\in[-1, 1] to the coefficent list of the corresponding Laurent polynomial

    inputs:
    clist: length n+1 array, the coefficeint list of even-power terms in the polynomial
    slist: length n+1 array, the coefficeint list of odd-power terms in the polynomial
    n: degree of the polynomial
    """
    czlist=np.append(np.append(np.flip(clist[1:])/2,  [clist[0]]),clist[1:]/2)
    szlist=np.append(np.append(-np.flip(slist[1:])/2j,  [slist[0]]),slist[1:]/2j)
    return czlist, szlist


# def GET_F(a, b, n):
#     '''mainly cut/paste from Fourier Approx file, should work okay. Takes the coefficient lists for a, b
#     and returns the coefficient list for $\mathcal{F}$'''
#     cz2list=LAUR_POLY_MULT(a, a)
#     sz2list=LAUR_POLY_MULT(b, b)
#     add1=np.append(np.append(np.zeros(2*n), 1), np.zeros(2*n))
#     calFc=add1-cz2list-sz2list
#     return calFc

def LAUR_POLY_MULT(coeffa, coeffb):
    """
    produces the coefficient list of the prodcut of polynomials a, b from their coefficient lists.
    inputs:
    coeffa, coeffb: length 2n+1 np arrays, the coefficient lists of two polynomials

    return: coefficient list of product polynomial
    """
    return sig.convolve(coeffa, coeffb, method='fft')

def LAUR_POLY_BUILD(coeff, n, z):
    """
    computes the float-point value of a Laurent polynomial from its coefficient list (does not assume symmetric coefficents).
    
    inputs:
    coef: length n+1 coefficient array
    n: float, degree of polynomial
    z: float or np array of points to check functional values

    return:
    float or np array of functional values
    """
    polyval=0
    for l in range(-n,n+1):
        polyval=polyval+coeff[l+n]*z**l
    return polyval

def POLY_BUILD(coeff, n, z):
    """computes the float-point value of a polynomial from its coefficient list.
   inputs:
    coef: length n+1 coefficient array
    n: float, degree of polynomial
    z: float or np array of points to check functional values

    return:
    float or np array of functional values
    """
    polyval=0
    for l in range(0,n+1):
        polyval=polyval+coeff[l]*z**l
        
    return polyval
