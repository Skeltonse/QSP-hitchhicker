"""
Generates data and plots for random polynomials
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
import time
import pickle

'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
from functions.matrix_inverse import CHEBY_INV_COEFF_ARRAY
import solvers.Wilson_method as wm
from simulators.projector_calcs import BUILD_PLIST, Ep_CALL, Ep_PLOT, SU2_CHECK
from simulators.angle_calcs import PROJ_TO_ANGLE, Wx_TO_R, W_PLOT, PAULI_CHECK, W_CALL, HAAHR_PLOT
from simulators.qet_sim import COMPLEX_QET_SIM, COMPLEX_QET_SIM2, QET_MMNT
#from scipy.linalg import expm
import simulators.matrix_fcns as mf
import simulators.unitary_calcs as uc
import parameter_finder as pf
import functions.random as frand

'''SPECIFIED BY THE USER - TT TIB ABOUT HOW THIS SHOULD LOOK'''
inst_tol=10**(-14)
pathname="random_benchmark_vary_nonzero.py"
ifsave=True
device='pc'
#t_array=np.linspace(400, 2000, 30, dtype=int)
#t_array=np.array([14, 16, 20], dtype=int)
#t_array=np.linspace(20, 1000, 15, dtype=int)
t_array=np.array([400])

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

fig, axes = plt.subplots(2, figsize=(12, 11))
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{braket}')

'''DEFINE THE FIGURE AND DOMAIN'''
plt.rcParams['font.size'] = 12
fsz=14
pts=20
theta=np.linspace(-np.pi,np.pi,pts)
xdata=np.cos(theta)

'''DEFINE PATHS FOR FILES'''
current_path = os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
if device=='mac':
    save_path=os.path.join(coeff_path,"benchmark_data/" )
else:
    save_path=os.path.join(coeff_path,"benchmark_data")

def RANDOM_FCN_CHECK(czlist, szlist, n, data, xdata):
    """
    Checks the polynomial approximation:
    Plots the polynomial expansion
    Prints a warning if czlist, szlist do not build real-on-circle polynomials within the instance tolerance
    
    inputs:
    czlist, szlist: 2n+1 length np arrays
    n: float
    tau: float, the simulation time
    data, xdata: np arrays of equal length. data points in \theta, x to compute and plot the functions
    subnorm optional subnormalization on exp(\tau x)
    """
    fig2=plt.figure()
    fl=lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+1j*lpf.LAUR_POLY_BUILD(szlist, n, np.exp(1j*data))
    plt.plot(data, np.real(fl), )  #label=r'$\mathcal{A}_{real}$'+'n is '+str(n), marker='.', 
    plt.plot(data, np.imag(fl),) #label=r'$\mathcal{A}_{imag}$'+'n is '+str(n)   marker='.', 
    plt.title("verify the approximation is alright")
    plt.legend()
    lpf.REAL_CHECK(czlist, n, theta=data, tol=inst_tol,fcnname='czlist')
    lpf.REAL_CHECK(szlist, n,theta=data,  tol=inst_tol, fcnname='szlist')
    plt.show()

def RUN_RANDOM_INSTANCES(nnz_array, n, ifsave=False):
    """
    Computes QSP parameters for each instance.
    Specific to random polynomials:
    nz is chosen as the maximum number of coefficents

    inputs:
    t_array: np array of degrees so ns kind of redundant
    ifsave: True/False command determining whether to save parameter values
    
    Output: dictionary with keys for each instance, and keys for arrays with the following:
    solution time,
    number of NR iterations to the solution
    polynomial degree
    approximation precision
    approximation norm
    """
    ###GENERATE ARRAYS TO SAVE RELEVANT DATA###
    AllInstDict={}
    DictLabels=t_array.astype(str)
    times=np.zeros([len(nnz_array)])
    iters=np.zeros([len(nnz_array)])
    nnz=np.zeros([len(nnz_array)])
    norms=np.zeros([len(nnz_array)])
    epsis=np.zeros([len(nnz_array)])

    ###MAIN LOOP###
    for nzind, nz in enumerate(nnz_array):
        print("possible nonzero coeffs", nz)
        tclist, tslist, nz=frand.RAND_JUMBLE_DECAY_CHEBY_GENERATOR(n, nz)
        print("actual nonzero  coeffs", nz)
        tclist, tslist, tczlist,tszlist, epsiapprox=frand.GET_NORMED(tclist, tslist, n, subnorm=2)
        #RANDOM_FCN_CHECK(tczlist, tszlist, tn, theta, xdata)
        #fig, axes = plt.subplots(2, figsize=(12, 8))
        
        Plist, Qlist, E0, a, b, c, d, n, tDict=pf.PARAMETER_FIND(tczlist, tszlist, n, theta, epsi=inst_tol, tDict={'nz':nz}, plots=False)
        
        times[nzind]=tDict['soltime']
        iters[nzind]=tDict['solit']
        nnz[nzind]=nz
        epsis[nzind]=epsiapprox
        AllInstDict[str(n)]=tDict
        fcnvals=lpf.LAUR_POLY_BUILD(a, n, np.exp(1j*theta))+1j*lpf.LAUR_POLY_BUILD(b, n, np.exp(1j*theta))
        norms[nzind]=pf.NORM_EXTRACT_FROMP(Plist, Qlist, E0, a, b, n,fcnvals, theta)

    AllInstDict['alltimes']=times
    AllInstDict['allits']=iters
    AllInstDict['allnonzero']=nnz
    AllInstDict['norms']=norms
    AllInstDict['epsiapprox']=epsis

    if ifsave==True:
        with open(save_path+"decayrandom_benchmark_data_to_"+str(t_array[-1])+".csv", 'wb') as f:
            pickle.dump(AllInstDict, f)
            
    return AllInstDict


def RANDOM_INSTANCE_PLOTS(ns, norms, iters, ifsave=False,  plotobj='NRits'):
    """
    Plots each instance against success measures
    option to plo the number of NR iterations or the solution time on the y-axis (default is NR)
    
    inputs:
    ns: np array with each iteration, usually the degree of the polynomial
    norms: np array with the max error in each instance
    iters: np. array with the number of NR iterations (or the solution time)
    ifsave: determines whether to save the plot or isplay it
    plotobj: changes plot labels for NR itertion or solution time
    
    """
    ####NORM PLOT ON LOG-LOG SCALE###
    axes[0].scatter(np.log10(ns), np.log10(norms))
    axes[0].set_ylabel(r'$\log_{10}||U_{QSP}-f||_{\infty}$',fontsize=fsz)
    axes[0].set_xlabel(r'$\log_{10}(n_{nz}),$ number of nonzero coefficients on a log scale', fontsize=fsz)
    axes[0].set_title('Max Error in QSP Value')

    ###ITERATIONS PLOT###
    axes[1].set_xlabel(r'Polynomial coefficients $nz$', fontsize=fsz)
    axes[1].plot(ns, iters, color='blue',marker='1')
    if plotobj=="NRits":
        axes[1].set_ylabel(r'number of Newton-Ralphson iterations')
        axes[1].set_title('Newton iterations vs polynomial degree')
        
    else:
        axes[1].set_ylabel(r'Completion step time ($s$)', fontsize=fsz)
        axes[1].set_title('solution time vs number of positive coefficients')

    plt.title('Random Polynomials')
    if ifsave==True:
        plt.savefig(save_path+"decayrandom_scalingplot_to_"+str(ns[-1])+".pdf")
    else:
        plt.show()
    return 

def NORM_FIT_PLOTS(ns, norms, epsiapprox,ifsave=False):
    """
    Computes best fit scaling for the error against the polynomial degree
    
    inputs:
    ns: np array with each iteration, usually the degree of the polynomial
    norms: np array with the max error in each instance
    epsiapprox: a numpy array with the target precision of each instance
    ifsave: True/False value, determines whether to save the plot or display it
    
    """
    ###GET THE CURVE FIT###
    whole_fit = curve_fit(pf.LS_FIT, xdata = np.log10(ns), ydata =np.log10(norms) )
    
    alpha=whole_fit[0]
    print('parameter standard deviations for linear fcn', np.sqrt(np.diag(whole_fit[1])))
    print('exponent and shift for log fit', (whole_fit[0]))

    ###GENERATE PLOTS###
    fig, axes=plt.subplots(figsize=(6, 6))
    axes.plot(np.log10(ns), alpha[0]*np.log10(ns) + alpha[1], 'r', label=str(np.around(alpha[0], 2))+r'$x+$'+str(np.around(alpha[1], 2)))
    axes.scatter(np.log10(ns), np.log10(norms), label='norm data')
    axes.set_ylabel(r'$\log_{10}\left(||U_{QSP}-f||_{\infty}\right)$',fontsize=fsz)
    axes.set_xlabel(r'$\log_{10}(n_{nz})$', fontsize=fsz)
    axes.set_title('Error in QSP Value on a log-log scale')
    
    axes.plot(np.log10(t_array), np.log10(epsiapprox), label=r'$\epsilon_{approx}$')
    axes.plot(np.log10(t_array), np.log10(8*n**2*epsiapprox))
    #axes.text(np.log10(ns[np.int64(len(ns)-1)])/2-np.log10(ns[np.int64(len(ns)-2)])/2, np.log10(alpha[0]*ns[-1] + alpha[1]),fontsize=22)
    
    plt.legend()
    
    plt.title('Random Polynomials')
    if ifsave==True:
        plt.savefig(save_path+"decayrandom_fittingplot_to_"+str(ns[-1])+".pdf")
    else:
        plt.show()
    return

nnz_array=np.array([5, 15, 25, 35, 45, 55, 65, 75, 85,95])

AllInstDict=RUN_RANDOM_INSTANCES(nnz_array, t_array[0])
RANDOM_INSTANCE_PLOTS(AllInstDict["allnonzero"], AllInstDict['norms'], AllInstDict['allits'], ifsave=ifsave)
#NORM_FIT_PLOTS(nnz_array, AllInstDict['norms'],AllInstDict['epsiapprox'], ifsave=ifsave)

