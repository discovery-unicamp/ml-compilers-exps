import numpy as np

from jax.scipy.signal import (
    correlate as cor,
    correlate2d as cor2d,
    convolve as conv,
    convolve2d as conv2d,
)

weights_1d = np.array([-0.5, 0.0, 0.5])
weights_2d = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1 , -1]
])

def convolve1d_fft(arr):
    return conv(arr, weights_1d, mode="same", method="fft")

def convolve1d_direct(arr):
    return conv(arr, weights_1d, mode="same", method="direct")
    
def correlate1d_fft(arr):
    return cor(arr, weights_1d, mode="same", method="fft")

def correlate1d_direct(arr):
    return cor(arr, weights_1d, mode="same", method="direct")

def convolve2d_fft(arr):
    return conv2d(arr, weights_2d, mode="same")

def convolve2d_direct(arr):
    return conv2d(arr, weights_2d, mode="same")
    
def correlate2d_fft(arr):
    return cor2d(arr, weights_2d, mode="same")

def correlate2d_direct(arr):
    return cor2d(arr, weights_2d, mode="same")
    
