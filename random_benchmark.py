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

import tikzplotlib

'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import solvers.Wilson_method as wm
from simulators.projector_calcs import BUILD_PLIST, Ep_CALL, Ep_PLOT, SU2_CHECK
import simulators.matrix_fcns as mf
import parameter_finder as pf
import functions.random as frand

'''SPECIFIED BY THE USER - TT TIB ABOUT HOW THIS SHOULD LOOK'''
inst_tol=10**(-12)
pathname="random_benchmark.py"
ifsave=True
#t_array=np.linspace(400, 2000, 30, dtype=int)
# t_array=np.array([14, 16, 20], dtype=int)
# t_array=np.linspace(20, 1000, 20, dtype=int)
# t_array=np.linspace(200, 2000, 10, dtype=int)
t_array=np.array([20,400])

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

'''DEFINE THE FIGURE AND DOMAIN'''
plt.rcParams['font.size'] = 12
fsz=14
pts=20
theta=np.linspace(-np.pi,np.pi,pts)
xdata=np.cos(theta)

'''DEFINE PATHS FOR FILES'''
current_path=os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
save_path=os.path.join(coeff_path,"benchmark_data")
save_path = os.path.normpath(save_path)

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

def RUN_RANDOM_INSTANCES(t_array, ifsave=False, ifsubplots=False):
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
    times=np.zeros([len(t_array)])
    iters=np.zeros([len(t_array)])
    ns=np.zeros([len(t_array)])
    norms=np.zeros([len(t_array)])
    epsis=np.zeros([len(t_array)])

    ###MAIN LOOP###
    for tind, tn in enumerate(t_array):
        nz=max(np.int64(tn/15), 5)#max(10, np.int32(tn/25))
        tclist, tslist, nz=frand.RAND_JUMBLE_DECAY_CHEBY_GENERATOR(tn, nz)
        
        tclist, tslist, tczlist,tszlist, epsiapprox=frand.GET_NORMED(tclist, tslist, tn, subnorm=(2))
        if ifsubplots==True:
            RANDOM_FCN_CHECK(tczlist, tszlist, tn, theta, xdata)
        Plist, Qlist, E0, a, b, c, d, n, tDict=pf.PARAMETER_FIND(tczlist, tszlist, tn, theta, epsi=inst_tol, tDict={'nz':nz}, plots=ifsubplots)

        times[tind]=tDict['soltime']
        iters[tind]=tDict['solit']
        ns[tind]=tDict['degree']
        epsis[tind]=epsiapprox
        AllInstDict[str(tn)]=tDict
        fcnvals=lpf.LAUR_POLY_BUILD(a, tn, np.exp(1j*theta))+1j*lpf.LAUR_POLY_BUILD(b, tn, np.exp(1j*theta))
        norms[tind]=pf.NORM_EXTRACT_FROMP(Plist, Qlist, E0, a, b, tn,fcnvals, theta)

    AllInstDict['alltimes']=times
    AllInstDict['allits']=iters
    AllInstDict['alldegrees']=ns
    AllInstDict['norms']=norms
    AllInstDict['epsiapprox']=epsis

    if ifsave==True:
        with open(os.path.join(save_path, "random_benchmark_data_to_"+str(t_array[-1])+".csv"), 'wb') as f:
            pickle.dump(AllInstDict, f)
            
    return AllInstDict


def RANDOM_INSTANCE_PLOTS(ns, norms, iters, ifsave=False,  plotobj='NRits'):
    """
    Plots each instance against success measures
    option to plot the number of NR iterations or the solution time on the y-axis (default is NR)
    
    inputs:
    ns: np array with each iteration, usually the degree of the polynomial
    norms: np array with the max error in each instance
    iters: np. array with the number of NR iterations (or the solution time)
    ifsave: determines whether to save the plot or isplay it
    plotobj: changes plot labels for NR itertion or solution time
    
    """
    fig, axes = plt.subplots(2, figsize=(16, 16))
    ####FIT###
    whole_fit = curve_fit(pf.LS_FIT, xdata = np.log10(ns), ydata =np.log10(norms) )
    alpha=whole_fit[0]
    ####NORM PLOT ON LOG-LOG SCALE###
    axes[0].plot(np.log10(ns), alpha[0]*np.log10(ns) + alpha[1], 'r', label=str(np.around(alpha[0], 2))+r'$\log_{10}(n)$'+str(np.around(alpha[1], 2)))
    axes[0].scatter(np.log10(t_array), np.log10(norms))
    axes[0].set_ylabel(r'$\log_{10}||U_{QSP}-f||_{\infty}$',fontsize=fsz)
    axes[0].set_xlabel(r'$\log_{10}(n)$', fontsize=fsz)
    axes[0].set_title('Max Error in QSP Value')
    axes[0].legend()
    ###ITERATIONS PLOT###
    axes[1].set_xlabel(r'Polynomial degree $n$', fontsize=fsz)
    axes[1].plot(ns, iters, color='blue',marker='1')
    if plotobj=="NRits":
        axes[1].set_ylabel(r'number of Newton-Ralphson iterations')
        axes[1].set_title('Newton iterations vs polynomial degree')
        
    else:
        axes[1].set_ylabel(r'Completion step time ($s$)', fontsize=fsz)
        axes[1].set_title('solution time vs polynomial degree')
    
    if ifsave==True:
        plt.savefig(save_path+"random_scalingplot_to_"+str(t_array[-1])+".pdf")
    elif ifsave=="tikz":
        print("legend is", str(np.around(alpha[0], 2))+r'$\log_{10}(n)$'+str(np.around(alpha[1], 2)))
        tikzplotlib.save("randomtikz.tex", flavor="context")
        plt.show()
    else:
        plt.show()
    return 

# AllInstDict=RUN_RANDOM_INSTANCES(t_array, ifsave=ifsave, ifsubplots=False)
# RANDOM_INSTANCE_PLOTS(t_array, AllInstDict['norms'], AllInstDict['alltimes'], ifsave="tikz", plotobj='times')
