import numpy as np

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
import functions.random as frand

"""motlaugh and Wiebe inputs"""
import torch
from torch.nn.functional import conv1d, pad
from torch.fft import fft
from torchaudio.transforms import Convolve, FFTConvolve


'''SPECIFIED BY THE USER'''
inst_tol=10**(-14)
pathname="test-berntson.py"
ifsave=False
device='pc'
#t_array=np.sort(np.append(np.array([20, 50, 80, 110, 140, 170,  230]), np.append(np.linspace(200, 1000, 17, endpoint=True, dtype=int), np.array([1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]))))


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
        coeff_path=coeff_path=current_path.replace("/test-berntson.py", "")
    else:
        coeff_path=current_path.replace("\\test-berntson.py", "")
        
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


def complementary(poly, N):
    """
    from Berntson and  Sünderhauf 2024
    Algorithm 1 to compute the complementary polynomial
    Parameters:
    poly : length (d+1) vector of monomial coefficients of P(z)
    N int : size of the FFT, N >= (d+1)
    Returns:
    length (d+1) vector of monomial coefficients of Q(z)
    """
    # Pad P to FFT dimension N
    paddedPoly = torch.zeros(N, dtype=torch.complex128)
    paddedPoly[:poly.shape[0]] = poly
    # Evaluate P(omega) at roots of unity omega
    pEval = torch.fft.ifft(paddedPoly, norm="forward")
    # Compute log(1-|P(omega)|^2) at roots of unity omega
    theLog = torch.log(1-torch.square(torch.abs(pEval)))
    # Apply Fourier multiplier in Fourier space
    modes = torch.fft.fft(theLog, norm="forward")
    modes[0] *= 1/2 # Note modes are ordered differently in the text
    modes[N//2+1:] = 0
    theLog = torch.fft.ifft(modes, norm="forward")
    # Compute coefficients of Q
    coefs = torch.fft.fft(torch.exp(theLog), norm="forward")
    # Truncate to length of Q polynomial
    q = coefs[:poly.shape[0]]
    return q

device = torch.device( "cpu")

def objective_torch(x, P):
    """motlaugh and Wiebe code"""
    x.requires_grad = True

    real_part = x[:len(x) // 2]
    imag_part = x[len(x) // 2:]

    real_flip = torch.flip(real_part, dims=[0])
    imag_flip = torch.flip(-1*imag_part, dims=[0])

    conv_real_part = FFTConvolve("full").forward(real_part, real_flip)
    conv_imag_part = FFTConvolve("full").forward(imag_part, imag_flip)

    conv_real_imag = FFTConvolve("full").forward(real_part, imag_flip)
    conv_imag_real = FFTConvolve("full").forward(imag_part, real_flip)

    # Compute real and imaginary part of the convolution
    real_conv = conv_real_part - conv_imag_part
    imag_conv = conv_real_imag + conv_imag_real

    # Combine to form the complex result
    conv_result = torch.complex(real_conv, imag_conv)

    # Compute loss using squared distance function
    loss = torch.norm(P - conv_result)**2
    return loss

def complex_conv_by_flip_conj(x):
    """motlaugh and Wiebe code"""
    real_part = x.real
    imag_part = x.imag

    real_flip = torch.flip(real_part, dims=[0])
    imag_flip = torch.flip(-1*imag_part, dims=[0])

    conv_real_part = FFTConvolve("full").forward(real_part, real_flip)
    conv_imag_part = FFTConvolve("full").forward(imag_part, imag_flip)

    conv_real_imag = FFTConvolve("full").forward(real_part, imag_flip)
    conv_imag_real = FFTConvolve("full").forward(imag_part, real_flip)

    # Compute real and imaginary part of the convolution
    real_conv = conv_real_part - conv_imag_part
    imag_conv = conv_real_imag + conv_imag_real

    # Combine to form the complex result
    return torch.complex(real_conv, imag_conv)

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

    ffttimes=np.zeros([len(t_array)])
    fftlengths=np.zeros([len(t_array)])

    opttimes=np.zeros([len(t_array)])
    optiters=np.zeros([len(t_array)])

    iters=np.zeros([len(t_array)])
    ns=np.zeros([len(t_array)])
    norms=np.zeros([len(t_array)])
    fftnorms=np.zeros([len(t_array)])
    optnorms=np.zeros([len(t_array)])
    
    ###MAIN LOOP###
    for tind, tau in enumerate(t_array):
        ###run the 'get coeffs' stuff
        filename="hsim_coeffs_epsi_" + "1.0e-14" + "_t_" + str(tau) 

        tczlist, tszlist, tn=get_coeffs(filename)
        tczlist, tszlist=tczlist*np.sqrt(2), tszlist*np.sqrt(2)
        HS_FCN_CHECK(tczlist, tszlist, tn,tau, theta, xdata, subnorm=np.sqrt(2))
        Plist, Qlist, E0, a, b, c, d,tn,  tDict=pf.PARAMETER_FIND(tczlist, -1j*tszlist, tn, theta,epsi=inst_tol,tDict={'tau':tau})
        
        t0=time.perf_counter()
        delta=1-1/np.sqrt(2)
        rd=(1/(1-delta))**(1/len(a))
        coefferror=inst_tol/3/(len(a)+1)/(2*len(a)+1)
        fftlength=2/np.log(rd)*np.log(8*np.log(1/delta)/coefferror/(rd-1))
        Q=complementary(torch.from_numpy(a+1j*b), np.int64(fftlength))
        t1=time.perf_counter()
        ffttimes[tind]=t1-t0

        Ppoints=lpf.POLY_BUILD(a+1j*b, 2*tn, np.exp(1j*theta))
        Qpoints=lpf.POLY_BUILD(Q.numpy(), 2*tn, np.exp(1j*theta))
        fftnorm=np.abs(Ppoints)**2+np.abs(Qpoints)**2
        #print(np.shape(norms))
        fftnorms[tind]=1-min(fftnorm)

        fftlengths[tind]=fftlength
        times[tind]=tDict['soltime']
        iters[tind]=tDict['solit']
        ns[tind]=tDict['degree']
        
        ###optimization###
        poly=torch.from_numpy(a+1j*b)

        granularity = 2 ** 25
        P = pad(poly, (0, granularity - poly.shape[0]))
        ft = fft(P)
        conv_p_negative = complex_conv_by_flip_conj(poly)*-1
        conv_p_negative[poly.shape[0] - 1] = 1 - torch.norm(poly) ** 2
        # Initializing Q randomly to start with
        initial = torch.randn(poly.shape[0]*2, device=device, requires_grad=True)
        initial = (initial / torch.norm(initial)).clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS([initial], tolerance_change=1e-12, max_iter=10000)

        t0 = time.time()

        def closure():
            optimizer.zero_grad()
            loss = objective_torch(initial, conv_p_negative)
            loss.backward()
            return loss

        optimizer.step(closure)

        t1 = time.time()

        total = t1-t0
        opttimes[tind]=total
        #final_vals.append(closure().item())
        optiters[tind]=total=optimizer.state[optimizer._params[0]]['n_iter']
        optnorms[tind]=closure().item()
        

        AllInstDict[str(tn)]=tDict
        fcnvals=np.exp(1j*tau*np.cos(theta))/np.sqrt(2)#lpf.LAUR_POLY_BUILD(a, tn, np.exp(1j*theta))+1j*lpf.LAUR_POLY_BUILD(b, tn, np.exp(1j*theta))
        norms[tind]=pf.NORM_EXTRACT_FROMP(Plist, Qlist, E0,a, b, tn, fcnvals, theta)
        print('approx error', norms[tind])        

    AllInstDict['alltimes']=times
    AllInstDict['allffttimes']=ffttimes
    AllInstDict['allfftlengths']=fftlengths
    AllInstDict['allopttimes']=opttimes
    AllInstDict['alloptiters']=optiters
    AllInstDict['allits']=iters
    AllInstDict['alldegrees']=ns
    AllInstDict['norms']=norms
    AllInstDict['fftnorms']=fftnorms
    AllInstDict['optnorms']=optnorms
    AllInstDict['evolutiontime']=t_array

    if ifsave==True:
        with open(save_path+"HS_benchmark_data_to_tau_"+str(t_array[-1])+".csv", 'wb') as f:
            pickle.dump(AllInstDict, f)
    return AllInstDict


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

def RUN_RANDOM_INSTANCES(t_array, ifsave=False):
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

    ffttimes=np.zeros([len(t_array)])
    fftlengths=np.zeros([len(t_array)])

    opttimes=np.zeros([len(t_array)])
    optiters=np.zeros([len(t_array)])

    fftnorms=np.zeros([len(t_array)])
    optnorms=np.zeros([len(t_array)])

    ###MAIN LOOP###
    for tind, tn in enumerate(t_array):
        nz=np.int32(tn/25)#max(10, np.int32(tn/25))
        print("possible", nz)
        tclist, tslist, nz=frand.RAND_JUMBLE_DECAY_CHEBY_GENERATOR(tn, nz)
        print("actual", nz)
        tclist, tslist, tczlist,tszlist, epsiapprox=frand.GET_NORMED(tclist, tslist, tn, subnorm=(2))
        #RANDOM_FCN_CHECK(tczlist, tszlist, tn, theta, xdata)
        #fig, axes = plt.subplots(2, figsize=(12, 8))
        
        Plist, Qlist, E0, a, b, c, d, n, tDict=pf.PARAMETER_FIND(tczlist, tszlist, tn, theta, epsi=inst_tol, tDict={'nz':nz}, plots=False)
        

        t0=time.perf_counter()
        delta=1-1/(2)
        rd=(1/(1-delta))**(1/len(a))
        coefferror=inst_tol/3/(len(a)+1)/(2*len(a)+1)
        fftlength=2/np.log(rd)*np.log(8*np.log(1/delta)/coefferror/(rd-1))
        Q=complementary(torch.from_numpy(a+1j*b), np.int64(fftlength))
        t1=time.perf_counter()
        ffttimes[tind]=t1-t0

        Ppoints=lpf.POLY_BUILD(a+1j*b, 2*tn, np.exp(1j*theta))
        Qpoints=lpf.POLY_BUILD(Q.numpy(), 2*tn, np.exp(1j*theta))
        fftnorm=np.abs(Ppoints)**2+np.abs(Qpoints)**2
        #print(np.shape(norms))
        fftnorms[tind]=1-min(fftnorm)

        ####cahnge to how we calculate norms for mine - completion step only###
        norm=np.abs(lpf.LAUR_POLY_BUILD(a+1j*b, tn, np.exp(1j*theta)))**2+np.abs(lpf.LAUR_POLY_BUILD(c+1j*d, tn, np.exp(1j*theta)))**2
        #print(np.shape(norms))
        norms[tind]=1-min(norm)


        fftlengths[tind]=fftlength
        times[tind]=tDict['soltime']
        iters[tind]=tDict['solit']
        ns[tind]=tDict['degree']
        
        ###optimization###
        poly=torch.from_numpy(a+1j*b)

        granularity = 2 ** 25
        P = pad(poly, (0, granularity - poly.shape[0]))
        ft = fft(P)
        conv_p_negative = complex_conv_by_flip_conj(poly)*-1
        conv_p_negative[poly.shape[0] - 1] = 1 - torch.norm(poly) ** 2
        # Initializing Q randomly to start with
        initial = torch.randn(poly.shape[0]*2, device=device, requires_grad=True)
        initial = (initial / torch.norm(initial)).clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS([initial], tolerance_change=1e-12, max_iter=10000)

        t0 = time.time()

        def closure():
            optimizer.zero_grad()
            loss = objective_torch(initial, conv_p_negative)
            loss.backward()
            return loss

        optimizer.step(closure)

        t1 = time.time()

        total = t1-t0
        opttimes[tind]=total
        #final_vals.append(closure().item())
        optiters[tind]=total=optimizer.state[optimizer._params[0]]['n_iter']
        optnorms[tind]=closure().item()

        fftlengths[tind]=fftlength

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
    AllInstDict['allffttimes']=ffttimes
    AllInstDict['allfftlengths']=fftlengths
    AllInstDict['allopttimes']=opttimes
    AllInstDict['alloptiters']=optiters
    AllInstDict['fftnorms']=fftnorms
    AllInstDict['optnorms']=optnorms

    if ifsave==True:
        with open(save_path+"random_benchmark_data_to_"+str(t_array[-1])+".csv", 'wb') as f:
            pickle.dump(AllInstDict, f)
            
    return AllInstDict


#t_array=np.linspace(20, 100, 1, dtype=int)
#t_array=np.array([1700, 1800, 1900, 2000])
t_array=np.array([20, 50, 80, 110, 140, 170, 200,  230])
#t_array=np.sort(np.append(np.array([20, 50, 80, 110, 140, 170,  230]), np.linspace(200, 1000, 17, endpoint=True, dtype=int)))


AllInstDict=RUN_RANDOM_INSTANCES(t_array, ifsave=False)
colors = ['#E69F00', '#56B4E9', '#009E73']
line_styles = ['-', '--', ':']

plt.plot(AllInstDict['alldegrees'], AllInstDict['alltimes'], color=colors[0], linestyle=line_styles[0], label="NR, avg error "+f"{np.average(AllInstDict['norms']):.2e}")
plt.plot(AllInstDict['alldegrees'], AllInstDict['allffttimes'],  color=colors[1], linestyle=line_styles[1],label="FFT, avg error "+f"{np.average(AllInstDict['fftnorms']):.2e}")
plt.plot(AllInstDict['alldegrees'], AllInstDict['allopttimes'],  color=colors[2], linestyle=line_styles[2],label="opt, avg error "+f"{np.average(AllInstDict['optnorms']):.2e}")
plt.legend()
plt.xlabel("degree")
plt.ylabel("completion step time")
plt.show()