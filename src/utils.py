from copy import deepcopy
import numpy as np

weights = {
    "1d": np.array([-0.5, 0.0, 0.5]),
    "2d": np.array([[-1, -1, -1], [-1, 8, 1], [1, 1, 1]]),
    "3d": np.array(
        [
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, 26, 1], [1, 1, 1]],
            [[-1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ]
    ),
}

weights_tvm = {
    "1d": np.array([-0.5, 0.0, 0.5]).reshape(1, 1, 3),
    "2d": np.array([[-1, -1, 1], [-1, 8, 1], [-1, 1, 1]]).reshape(1, 3, 3),
    "3d": np.array(
        [
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, 1]],
            [[-1, -1, -1], [-1, 26, 1], [1, 1, 1]],
            [[-1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ]
    ),
}

dataset_class = {
    "32-32-32": 1,
    "64-64-64": 1,
    "128-128-128": 1,
    "64-64-256": 2,
    "64-64-512": 2,
    "128-128-512": 2,
    "128-128-1024": 2,
}

attr_class = {
    "fft": [2],
    "hilbert": [2],
    "envelope": [2],
    "inst-phase": [2],
    "cos-inst-phase": [2],
    "relative-amplitude-change": [2],
    "amplitude-acceleration": [2],
    "inst-frequency": [2],
    "inst-bandwidth": [2],
    "dominant-frequency": [2],
    "frequency-change": [2],
    "sweetness": [2],
    "quality-factor": [2],
    "response-phase": [2],
    "response-frequency": [2],
    "response-amplitude": [2],
    "apparent-polarity": [2],
    "convolve1d": [1],
    "correlate1d": [1],
    "convolve2d": [1],
    "correlate2d": [1],
    "convolve3d": [1],
    "correlate3d": [1],
    "glcm-asm": [1],
    "glcm-contrast": [1],
    "glcm-correlation": [1],
    "glcm-variance": [1],
    "glcm-energy": [1],
    "glcm-entropy": [1],
    "glcm-mean": [1],
    "glcm-std": [1],
    "glcm-dissimilarity": [1],
    "glcm-homogeneity": [1],
}


def check_attr_dataset_match(attr, dataset):
    return dataset_class[dataset] in attr_class[attr]


def extract_data(data, name):
    if "1d" in name:
        il, xl, _ = data.shape
        data = deepcopy(data[il // 2, xl // 2, :])
    elif "2d" in name:
        il, _, _ = data.shape
        data = deepcopy(data[il // 2, :, :])
    return data


def extract_data_tvm(data, name):
    if "1d" in name:
        il, xl, t = data.shape
        data = deepcopy(data[il // 2, xl // 2, :].reshape(1, 1, t))
    elif "2d" in name:
        il, xl, t = data.shape
        data = deepcopy(data[il // 2, :, :].reshape(1, xl, t))
    return data
