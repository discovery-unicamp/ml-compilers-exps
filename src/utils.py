from copy import deepcopy
import numpy as np

weights = {
    "1d": np.array([-0.5, 0.0, 0.5]),
    "2d": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "3d": np.array(
        [
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, 26, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        ]
    ),
}

weights_tvm = {
    "1d": np.array([-0.5, 0.0, 0.5]).reshape(1, 1, 3),
    "2d": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).reshape(1, 3, 3),
    "3d": np.array(
        [
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, 26, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        ]
    ),
}


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
