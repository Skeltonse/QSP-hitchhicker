from nlft_qsp.poly import Polynomial
from nlft_qsp.nlft import NonLinearFourierSequence
from nlft_qsp.rand import random_polynomial

from nlft_qsp.solvers import convolve_optimize, weiss, prony, layer_stripping, riemann_hilbert, half_cholesky, nlfft

import time

import matplotlib.pyplot as plt
n=1
bcoeffs=[0,0.4, 0]
b=Polynomial(bcoeffs)
# b = random_polynomial(n+1, eta=0.5)
a, c = weiss.ratio(b)

completion_err = (a * a.conjugate() + b * b.conjugate() - 1).l2_norm()
print(b, completion_err)