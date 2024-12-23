import torch
from torch.fft import fft

def complementary(poly, N):
    """
    code copied from Berntson and  Sünderhauf 2024 (Algorithm 1)
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

