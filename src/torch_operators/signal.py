import torch.nn.functional as F
import warnings


def convolve1d(arr, w):
    return F.conv1d(arr.unsqueeze(0).unsqueeze(0), w.flip(0).unsqueeze(0).unsqueeze(0), padding="same").squeeze()


def correlate1d(arr, w):
    return F.conv1d(arr.unsqueeze(0).unsqueeze(0), w.unsqueeze(0).unsqueeze(0), padding="same").squeeze()


def convolve2d(arr, w):
    return F.conv2d(arr.unsqueeze(0).unsqueeze(0), w.flip([0, 1]).unsqueeze(0).unsqueeze(0), padding="same").squeeze()


def correlate2d(arr, w):
    return F.conv2d(arr.unsqueeze(0).unsqueeze(0), w.unsqueeze(0).unsqueeze(0), padding="same").squeeze()


def convolve3d(arr, w):
    return F.conv3d(arr.unsqueeze(0).unsqueeze(0), w.flip([0, 1, 2]).unsqueeze(0).unsqueeze(0), padding="same").squeeze()


def correlate3d(arr, w):
    return F.conv3d(arr.unsqueeze(0).unsqueeze(0), w.unsqueeze(0).unsqueeze(0), padding="same").squeeze()
