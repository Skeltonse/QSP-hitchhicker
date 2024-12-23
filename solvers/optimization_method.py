import torch
from torchaudio.transforms import Convolve, FFTConvolve

"""Code from Motlaugh and Wiebe 2024"""
def objective_torch(x, P):
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