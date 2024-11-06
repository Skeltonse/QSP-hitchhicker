"""
Generates data and plots for truncated Jacobi-Angler expansions
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
import simulators.matrix_fcns as mf
import simulators.unitary_calcs as uc
import parameter_finder as pf

'''SPECIFIED BY THE USER'''
inst_tol=10**(-14)
pathname="HS_benchmark.py"
ifsave=False
device='pc'
#t_array=np.sort(np.append(np.array([20, 50, 80, 110, 140, 170,  230]), np.append(np.linspace(200, 1000, 17, endpoint=True, dtype=int), np.array([1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]))))
#t_array=np.array([1700, 1800, 1900, 2000])
t_array=np.array([20, 50, 80, 110, 140, 170, 200,  230])
#t_array=np.sort(np.append(np.array([20, 50, 80, 110, 140, 170,  230]), np.linspace(200, 1000, 17, endpoint=True, dtype=int)))


'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
#plt.rcParams.update({'text.usetex': True,'font.family': 'serif',})
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{braket}')

'''DEFINE THE FIGURE AND DOMAIN'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


plt.rcParams['font.size'] = 12
fsz=14
pts=100
theta=np.linspace(-np.pi,np.pi,pts)
xdata=np.cos(theta)

'''DEFINE PATHS FOR FILES'''
if device=='mac':
    current_path = os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"benchmark_data/")
else:
    current_path = os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"benchmark_data\\")


def get_coeffs(filename):
    """
    Extracts coefficient lists for Laurent polynomials from a csv file
    input:
    filename: string specifying the coefficient file
    output:
    ccoeff, scoeff: numpy arrays. the coefficient lists of respectively reciprocal and anti-reciprocal Laurent polynomials
    n: the max degree of the Laurent polynomials
    """
    current_path = os.path.abspath(__file__)
    if device=='mac':
        coeff_path=coeff_path=current_path.replace("/HS_benchmark.py", "")
    else:
        coeff_path=current_path.replace("\HS_benchmark.py", "")
        
    abs_path=os.path.join(coeff_path,"csv_files", filename+".csv")
    with open(abs_path, 'r') as file:
        csv_reader = csv.reader(file)
        column1=[]
        column2=[]
        column3=[]
        next(csv_reader)

        for row in csv_reader:
            col1, col2, col3= row[:3]  # Unpack the first three columns
            column1.append(col1)
            column2.append(col2)
            column3.append(col3)
    ###I keep imaginary numbers in the coeff arrays so each array will produce a real-on-circle poly without pain
    ccoeff=np.array(column1).astype(float)
    scoeff=1j*np.array(column2).astype(float)
    n=int(column3[0])
    return ccoeff, scoeff, n

def HS_FCN_CHECK(czlist, szlist, n, tau, data, xdata, subnorm=1, inst_tol=inst_tol, plots=False):
    """
    Checks the polynomial approximation:
    Plots the polynomial expansion against the target function exp(\tau x)
    Prints a warning if czlist, szlist do not build real-on-circle polynomials within the instance tolerance
    
    inputs:
    czlist, szlist: 2n+1 length np arrays
    n: float
    tau: float, the simulation time
    data, xdata: np arrays of equal length. data points in \theta, x to compute and plot the functions
    subnorm optional subnormalization on exp(\tau x)
    inst_tol: float, the precision of the approximation
    plots: True/False determines whether the plot is displayed
    """
    ####COMPUTE FUNCTION AND APPROXIMATION VALUES AT EACH POINT###
    fl=lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))
    targetfcn=np.exp(1j*xdata*tau)/subnorm

    ###PLOT THE APPROXIMATIONS###
    if plots==True:
        fig, axes = plt.subplots(1, )
        axes.plot(xdata, np.real(fl), marker='.', label=r'$\mathcal{A}_{real}$')  
        axes.plot(xdata, np.real(targetfcn),label=r'$e^{it\lambda}_{real}$', )
        axes.plot(xdata, np.imag(fl), marker='.', label=r'$\mathcal{A}_{imag}$')  
        axes.plot(xdata, np.imag(targetfcn),label=r'$e^{it\lambda}_{imag}$', )
        axes.set_title("verify the approximation is alright")
        axes.legend()
        plt.show()
    
    ###CHECK THE PROPERTIES OF EACH FUNCTION AND THE DISTANCE BETWEEN APPROXIMATIONS###
    lpf.REAL_CHECK(czlist, n, theta=data, tol=inst_tol,fcnname='czlist')
    lpf.REAL_CHECK(-1j*szlist, n,theta=data,  tol=inst_tol, fcnname='szlist')
    obj=fl-targetfcn
    normlist=np.sqrt(np.real(obj)**2+np.imag(obj)**2)
    if np.max(normlist)>inst_tol:
        print("Warning, polynmial is not an epsilon close approximation of exp(tau x)")
        print(np.max(normlist))

        
def RUN_HS_INSTANCES(t_array, ifsave=False, subnorm=1):
    """
    Computes QSP parameters for each instance
    Specific to J-A polynomials:
    t_array is the list of simulation times,
    all instances have the same precision \epsilon defined by inst_tol
    coefficients are imported with subnormalization 2, additional subnormalization can be handled with subnorm

    inputs:
    t_array: np array
    ifsave: True/False command determining whether to save parameter values
    subnorm: optional additional subnormalization
    
    Output: dictionary with keys for each instance, and keys for arrays with the following:
    solution time,
    number of NR iterations to the solution,
    polynomial degree,
    approximation norm,
    evolution time for each instance
    """
    ###GENERATE ARRAYS TO SAVE RELEVANT DATA###
    AllInstDict={}
    DictLabels=t_array.astype(str)
    times=np.zeros([len(t_array)])
    iters=np.zeros([len(t_array)])
    ns=np.zeros([len(t_array)])
    norms=np.zeros([len(t_array)])
    
    ###MAIN LOOP###
    for tind, tau in enumerate(t_array):
        ###run the 'get coeffs' stuff
        filename="hsim_coeffs_epsi_" + "1.0e-14" + "_t_" + str(tau) 

        tczlist, tszlist, tn=get_coeffs(filename)
        tczlist, tszlist=tczlist*np.sqrt(2), tszlist*np.sqrt(2)
        HS_FCN_CHECK(tczlist, tszlist, tn,tau, theta, xdata, subnorm=np.sqrt(2))
        Plist, Qlist, E0, a, b, c, d,tn,  tDict=pf.PARAMETER_FIND(tczlist, -1j*tszlist, tn, theta,epsi=inst_tol,tDict={'tau':tau})
        
        times[tind]=tDict['soltime']
        iters[tind]=tDict['solit']
        ns[tind]=tDict['degree']
        AllInstDict[str(tn)]=tDict
        fcnvals=np.exp(1j*tau*np.cos(theta))/np.sqrt(2)#lpf.LAUR_POLY_BUILD(a, tn, np.exp(1j*theta))+1j*lpf.LAUR_POLY_BUILD(b, tn, np.exp(1j*theta))
        norms[tind]=pf.NORM_EXTRACT_FROMP(Plist, Qlist, E0,a, b, tn, fcnvals, theta)
        print('approx error', norms[tind])        

    AllInstDict['alltimes']=times
    AllInstDict['allits']=iters
    AllInstDict['alldegrees']=ns
    AllInstDict['norms']=norms
    AllInstDict['evolutiontime']=t_array

    if ifsave==True:
        with open(save_path+"HS_benchmark_data_to_tau_"+str(t_array[-1])+".csv", 'wb') as f:
            pickle.dump(AllInstDict, f)
    return AllInstDict


def HS_INSTANCE_PLOTS(t_array, ns, norms, iters,  ifsave=False, plotobj='NRits', withLSF=False):
    """
    Plots each instance against success measures 
    option to plot the number of NR iterations or the solution time on the y-axis (default is NR)
    
    inputs:
    t_array: np array with the evolution time 
    ns: np array with the degree of the polynomial 
    norms: np array with the max error in each instance
    iters: np. array with the number of NR iterations (or the solution time)
    ifsave: determines whether to save the plot or display it
    plotobj: changes plot labels for NR itertion or solution time
    withLSF: option to add a least squares fit to the plot
    """
    fig, axes = plt.subplots(2, figsize=(12,10))
    ####NORM PLOT ON LOG-LOG SCALE###
    axes[0].scatter(np.log10(ns), np.log10(norms))
    axes[0].set_ylabel(r'$\log_{10}\left(||U_{QSP}-f||_{\infty}\right)$',fontsize=fsz)
    axes[0].set_xlabel(r'$\log_{10}(\tau)$', fontsize=fsz)
    axes[0].set_title('Error in QSP Value')
    
    if withLSF==True:
        secname="HS_scalingplot_wfit_to_"
        whole_fit = curve_fit(pf.LS_FIT, xdata = np.log10(ns), ydata =np.log10(norms) )
        alpha=whole_fit[0]
        print('parameter standard deviations for linear fcn', np.sqrt(np.diag(whole_fit[1])))
        axes[0].plot(np.log10(ns), alpha[0]*np.log10(ns) + alpha[1], 'r', label=str(np.around(alpha[0], 2))+r'$\log_{10}(n)$'+str(np.around(alpha[1], 2)))
        #axes[0].plot(np.log10(ns), alpha[0]*np.log10(ns), 'r', label=str(np.around(alpha[0], 2))+r'$\log_{10}(n)$')
        axes[0].legend()
    else:
        secname="HS_scalingplot_to_"
        
    axes[1].set_xlabel(r'Polynomial degree $n$', fontsize=fsz)
    if plotobj=="NRits":
        axes[1].set_ylabel(r'Newton-Raphson iterations')
        axes[1].set_title('Newton-Raphson iterations vs polynomial degree')
        axes[1].plot(ns, iters, color='blue',marker='1')
    else:
        axes[1].set_ylabel(r'Completion step time ($s$)', fontsize=fsz)
        axes[1].set_title('solution time vs polynomial degree')
        axes[1].plot(ns, iters, color='blue',marker='1')
        
    plt.legend()
    plt.suptitle('HS Polynomials')
    if ifsave==True:
        plt.savefig(save_path+secname+str(t_array[-1])+"wrtexpfcn.pdf")
    else:
        plt.show()
    return

#AllInstDict=RUN_HS_INSTANCES(t_array, ifsave=True)
#print(AllInstDict['norms'])
#HS_INSTANCE_PLOTS(t_array,  AllInstDict['alldegrees'],AllInstDict['norms'], iters=AllInstDict['alltimes'], ifsave=ifsave, withLSF=False,  plotobj='times',)
