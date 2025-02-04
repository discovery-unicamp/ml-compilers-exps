import torch
import torch.nn.functional as F


def glcm_kernel(base, neighbour, levels):
    n = base.shape[0]
    level = torch.arange(n, device=base.device)
    result = base.new_zeros((n, levels, levels))
    inc = base.new_ones((1,))
    base = base.long()
    neighbour = neighbour.long() 
    result[level, base, neighbour] = inc
    result[level, neighbour, base] += inc
    result_normed = torch.sum(result, dim=0) / (2 * n)

    return result_normed


def glcm_compute_D0(window, levels):  # EAST
    base = window[:, :-1].flatten()
    neighbour = window[:, 1:].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_compute_D1(window, levels=16):  # SOUTH
    base = window[:-1, :].flatten()
    neighbour = window[1:, :].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_compute_D2(window, levels):  # SOUTH_EAST
    base = window[:-1, :-1].flatten()
    neighbour = window[1:, 1:].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_compute_D3(window, levels):  # SOUTH_WEST
    base = window[1:, -1].flatten()
    neighbour = window[:-1, 1:].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_base(X, levels=16, direction=1, window_size=7):
    X_min = torch.min(X)
    X_max = torch.max(X)
    hw = window_size // 2
    gray_level = ((X - X_min) / (X_max - X_min)) * (levels - 1)
    windows = F.unfold(gray_level.unsqueeze(1), (window_size, window_size), padding=hw)
    windows = windows.transpose(1, 2)
    windows = windows.reshape(-1, window_size, window_size)
    if direction == 0:
        f = torch.vmap(glcm_compute_D0, in_dims=0, chunk_size=32768)
    elif direction == 1:
        f = torch.vmap(glcm_compute_D1, in_dims=0, chunk_size=32768)
    elif direction == 2:
        f = torch.vmap(glcm_compute_D2, in_dims=0, chunk_size=32768)
    elif direction == 3:
        f = torch.vmap(glcm_compute_D3, in_dims=0, chunk_size=32768)
    return f(windows, levels=levels).reshape(*X.shape, levels, levels)


def glcm_dissimilarity(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I, J = torch.meshgrid(torch.arange(16, device=X.device), torch.arange(16, device=X.device), indexing="ij")
    weights = torch.abs(I - J)
    return torch.sum(weights * glcm, dim=(-2, -1))


def glcm_homogeneity(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I, J = torch.meshgrid(torch.arange(16, device=X.device), torch.arange(16, device=X.device), indexing="ij")
    weights = 1.0 / (1.0 + (I - J) ** 2)
    return torch.sum(weights * glcm, dim=(-2, -1))


def glcm_contrast(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I, J = torch.meshgrid(torch.arange(16, device=X.device), torch.arange(16, device=X.device), indexing="ij")
    weights = (I - J) ** 2
    return torch.sum(weights * glcm, dim=(-2, -1))


def glcm_energy(X, levels=16, direction=1):
    results = torch.sqrt(glcm_asm(X, levels, direction))
    return results


def glcm_asm(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    results = torch.sum(glcm**2, dim=(-2, -1))
    return results


def glcm_correlation(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    J, I = torch.meshgrid(torch.arange(16, device=X.device), torch.arange(16, device=X.device), indexing="ij")
    I = I.reshape(1, 1, 1, *I.shape)
    J = J.reshape(1, 1, 1, *J.shape)
    mean = torch.sum(I * glcm, dim=(-2, -1))
    mean = mean.reshape(*mean.shape, 1, 1)
    var = torch.sum(glcm * ((I - mean) ** 2), axis=(-2, -1))
    var = var.reshape(*var.shape, 1, 1)
    var_wo_0s = torch.where(var == 0, 1, var)
    return torch.sum((glcm * (((I - mean) * (J - mean)) / var_wo_0s)), axis=(-2, -1))


def glcm_mean(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I = torch.arange(levels, device=X.device).reshape(1, 1, 1, levels, 1)
    mean = torch.sum(I * glcm, axis=(-2, -1))
    return mean


def glcm_variance(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I = torch.arange(levels, device=X.device).reshape(1, 1, 1, 1, levels)
    mean = torch.sum(I * glcm, dim=(-2, -1))
    mean = mean.reshape(*mean.shape, 1, 1)
    return torch.sum(glcm * ((I - mean) ** 2), axis=(-2, -1))


def glcm_std(X, levels=16, direction=1):
    variance = glcm_variance(X, levels, direction)
    return torch.sqrt(variance)


def glcm_entropy(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    glcm_wo_0s = torch.where(glcm == 0, 1, glcm)
    log = -torch.log(glcm_wo_0s)
    results = torch.sum(glcm * log, dim=(-2, -1))
    return results
