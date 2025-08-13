from jax.scipy.signal import (
    correlate as cor,
    correlate2d as cor2d,
    convolve as conv,
    convolve2d as conv2d,
)


def convolve1d(arr, w):
    return conv(arr, w, mode="same", method="direct")


def correlate1d(arr, w):
    return cor(arr, w, mode="same", method="direct")


def convolve2d(arr, w):
    return conv2d(arr, w, mode="same")


def correlate2d(arr, w):
    return cor2d(arr, w, mode="same")


def convolve3d(arr, w):
    return conv(arr, w, mode="same", method="direct")


def correlate3d(arr, w):
    return cor(arr, w, mode="same", method="direct")
