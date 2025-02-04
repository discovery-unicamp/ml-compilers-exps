import torch
import numpy as np
from .utils import first_derivative


def fft(X):
    return torch.fft.fft(X, dim=-1)


def hilbert(X):
    _, _, z = X.shape
    freq = torch.fft.fft(X, dim=-1)
    h = X.new_zeros((z,), dtype=X.dtype)    
    if z % 2 == 0:
        h[0] = 1
        h[z // 2] = 1
        h[1 : z // 2] = 2    
    else:
        h[0] = 1
        h[1 : (z + 1) // 2] = 2
    h = torch.reshape(h, (1, 1, z))
    freq = freq * h
    imag = torch.imag(torch.fft.ifft(freq, dim=-1))
    return X + 1j * imag


def envelope(X):
    return torch.abs(hilbert(X))


def instantaneous_phase(X):
    return torch.rad2deg(torch.angle(hilbert(X)))


def cosine_instantaneous_phase(X):
    return torch.cos(torch.angle(hilbert(X)))


def relative_amplitude_change(X):
    env = envelope(X)
    env_prime = first_derivative(env)

    return torch.clip(env_prime / env, -1, 1)


def amplitude_acceleration(X):
    rac = relative_amplitude_change(X)
    return first_derivative(rac)


def instantaneous_bandwidth(X):
    rac = relative_amplitude_change(X)
    return torch.abs(rac) / (2.0 * np.pi)
