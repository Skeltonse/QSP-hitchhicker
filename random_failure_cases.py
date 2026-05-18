"""
Checks the failed random polynomials
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
import time
import pickle

import torch
from torch.nn.functional import conv1d, pad
from torch.fft import fft
import tikzplotlib

'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import solvers.Wilson_method as wm
from simulators.projector_calcs import BUILD_PLIST, Ep_CALL, Ep_PLOT, SU2_CHECK
import simulators.matrix_fcns as mf
import parameter_finder as pf
import functions.random as frand

from solvers.optimization_method import objective_torch, complex_conv_by_flip_conj
from solvers.FFT_method import complementary


from nlft_qsp.poly import Polynomial
from nlft_qsp.nlft import NonLinearFourierSequence
from nlft_qsp.solvers import convolve_optimize, weiss, riemann_hilbert, half_cholesky, nlfft



'''DEFINE PATHS FOR FILES'''
inst_tol=10**(-12)
pathname="random_failure_cases.py"

current_path=os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
save_path=os.path.join(coeff_path,"thesis_data")
save_path = os.path.normpath(save_path)
thesispath=os.path.normpath(os.path.join(coeff_path, "thesis_data"))

indexstringarray=range(10)
t_array=np.linspace(200, 2000, 20, dtype=int)

fsz=14
pts=200
theta=np.linspace(-np.pi,np.pi,pts)

# print(allinstsdict['200'][''])
faildict={}
failcount=0

succdict={}
succcount=0
##code a way to detect instances that failed and then save that dictionary element in a 'fail' dictionary

# with open(os.path.join(save_path, "random"+"_benchmark_data_to_"+str(2000)+".csv"), 'rb') as f:
#     allinstsdict=pickle.load(f)
# faileddegress=(allinstsdict['alldegrees'][np.where(allinstsdict['norms']>10**(-10))])
# succdegress=(allinstsdict['alldegrees'][np.where(allinstsdict['norms']<=10**(-10))])
# for it, val in enumerate(faileddegress):
#     faildict['failcount'+str(failcount)]=allinstsdict[str(int(val))]
#     failcount+=1
# for it, val in enumerate(succdegress):
#     succdict['succcount'+str(succcount)]=allinstsdict[str(int(val))]
#     succcount+=1

for j in range(0, 10):
    
    with open(os.path.join(save_path, "random"+str(indexstringarray[j])+"_benchmark_data_to_"+str(2000)+".csv"), 'rb') as f:        allinstsdict=pickle.load(f)
    faileddegress=(allinstsdict['alldegrees'][np.where(allinstsdict['norms']>10**(-10))])
    succdegress=(allinstsdict['alldegrees'][np.where(allinstsdict['norms']<=10**(-10))])

    for it, val in enumerate(faileddegress):
        faildict['failcount'+str(failcount)]=allinstsdict[str(int(val))]
        failcount+=1
    for it, val in enumerate(succdegress):
        succdict['succcount'+str(succcount)]=allinstsdict[str(int(val))]
        succcount+=1

print("successful instances", succcount)
print("failed instances", failcount)

###check alternate solution###
def FFT_METHOD(a, b, nearcoherent=False, realpoly=False):
    t0=time.perf_counter()

    if nearcoherent==False:
        delta=1-0.7
        
        rd=(1/(1-delta))**(1/len(a))
        coefferror=inst_tol/3/(len(a)+1)/(2*len(a)+1)
        fftlength=2/np.log(rd)*np.log(8*np.log(1/delta)/coefferror/(rd-1))
        Q=complementary(torch.from_numpy(a+1j*b), np.int64(fftlength))
        if realpoly==True:
            Q=complementary(torch.from_numpy(a), np.int64(fftlength))

        t1=time.perf_counter()

    if nearcoherent==True:
        scalingfactor=1-1*10**(-8)/4
        delta=10**(-8)/4 
        coefferror=10**(-10)/5/(len(a)+1)/(2*len(a)+1)#near-coherent precision
        aprime=np.power(scalingfactor*(a),nz)
        bprime=np.power(scalingfactor*b,nz)
        fftlength=1990000000
        Q=complementary(torch.from_numpy(aprime+1j*bprime), np.int64(fftlength))
        t1=time.perf_counter()

    
    ffttimes=t1-t0

    Ppoints=lpf.POLY_BUILD(a+1j*b, 2*nz, np.exp(1j*theta))
    if realpoly==True:
        Ppoints=lpf.POLY_BUILD(a, 2*nz, np.exp(1j*theta))
    Qpoints=lpf.POLY_BUILD(Q.numpy(), 2*nz, np.exp(1j*theta))
    fftnorm=np.abs(Ppoints)**2+np.abs(Qpoints)**2
            
    fftnorms=1-min(fftnorm)
    print("BernSuen solution quality", fftnorms)
    return fftnorms


#### Weiss method###
def WEISS_METHOD(a, b):

    t0 = time.time()
    bPoly=Polynomial(a+1j*b)
    aPoly, cPoly = weiss.ratio(bPoly)
    completion_err = (aPoly * aPoly.conjugate() + bPoly * bPoly.conjugate() - 1).l2_norm()
    t1 = time.time()
    weissnorms=completion_err
    weisstimes=t1-t0 
    print("Weiss solution quality", weissnorms)
    # print("Weiss solution time",  weisstimes)
    # print("Weiss solution coefficients", aPoly)
    return weissnorms


nfailarry=np.array([])
nsuccarry=np.array([])
normfailarry=np.array([])
normsuccarry=np.array([])

fftarray=np.array([])
weissarray=np.array([])
wilsonarray=np.array([])

for j in range(failcount):
    a=faildict['failcount'+str(j)]['a']
    b=faildict['failcount'+str(j)]['b']
    nz=faildict['failcount'+str(j)]['degree']
    
    nfailarry=np.append(nfailarry, nz)
   
    maxnorm=max(abs(lpf.LAUR_POLY_BUILD(a+1j*b, int((len(a)-1)/2),  np.exp(1j*theta))))
    normfailarry=np.append(normfailarry, maxnorm)
    
    ##check if other methods fail##
    # print(FFT_METHOD(a, b))
    # print(WEISS_METHOD(a, b))

    
    fftarray=np.append(fftarray, FFT_METHOD(a/2, b/2))
    weissarray=np.append(weissarray, WEISS_METHOD(a/2, b/2))

    ###does my method work with an additional subnormalaization?
    Plist, Qlist, E0, a, b, c, d, n, tDict=pf.PARAMETER_FIND(a/2, b/2, int((len(a)-1)/2), theta, epsi=inst_tol, tDict={'nz':nz}, plots=False)
    norm=np.abs(lpf.LAUR_POLY_BUILD(a+1j*b, int((len(a)-1)/2), np.exp(1j*theta)))**2+np.abs(lpf.LAUR_POLY_BUILD(c+1j*d, int((len(a)-1)/2), np.exp(1j*theta)))**2
    wilsonarray=np.append(wilsonarray, 1-min(norm))

colors = ['#E69F00', '#56B4E9', '#009E73']
line_styles = ['-', '--', ':']

nfailarry=nfailarry[np.argsort(nfailarry)]
wilsonarray=wilsonarray[np.argsort(nfailarry)]
weissarray=weissarray[np.argsort(nfailarry)]
fftarray=fftarray[np.argsort(nfailarry)]

for j in range(succcount):
    a=succdict['succcount'+str(j)]['a']
    b=succdict['succcount'+str(j)]['b']
    nz=succdict['succcount'+str(j)]['degree']
    

    nsuccarry=np.append(nsuccarry,nz)

    maxnorm=max(abs(lpf.LAUR_POLY_BUILD(a+1j*b, nz,  np.exp(1j*theta))))
    normsuccarry=np.append(normsuccarry, maxnorm)



plt.plot(nfailarry, np.log10(wilsonarray), color=colors[0],  label="Wilson, avg error "+f"{np.average(wilsonarray):.2e}")
plt.plot(nfailarry, np.log10(fftarray),  color=colors[1],label="FFT, avg error "+f"{np.average(fftarray):.2e}")
# plt.plot(nfailarry, AllInstDict['allopttimes'],  color=colors[2], label="opt, avg error "+f"{np.average(AllInstDict['optnorms']):.2e}")
plt.plot(nfailarry, np.log10(weissarray),  color=colors[2], label="weiss, avg error "+f"{np.average(weissarray):.2e}")
plt.xlabel("degree")
plt.ylabel("norm")
plt.title("subnormalized instances")
#tikzplotlib.save(os.path.join(thesispath, "succcomp.tex"), flavor="context")
plt.show()
print("the norm list", normfailarry)



plt.scatter(nfailarry/(2*nz+1), np.ones(len(nfailarry)))
plt.plot(nsuccarry/(2*nz+1), np.ones(len(nsuccarry)))
plt.xlabel("degree")
tikzplotlib.save(os.path.join(thesispath, "sparsefaildegreelineplot.tex"), flavor="context")
plt.show()

plt.scatter(nfailarry, normfailarry)
plt.scatter(nsuccarry, normsuccarry)
plt.xlabel("degree")
plt.ylabel("polynomial norm")
#tikzplotlib.save(os.path.join(thesispath, "sparsefailnormplot.tex"), flavor="context")
plt.show()
