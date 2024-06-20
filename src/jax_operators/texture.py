import jax.numpy as jnp
import numpy as np

import jax
import jax.numpy as jnp


def moving_window(matrix, window_shape):
    # implemenation based on Gintasz answer at https://github.com/google/jax/issues/3171
    x, y, z = matrix.shape
    w_x, w_y, w_z = window_shape

    startsx = jnp.arange(x - w_x + 1)
    startsy = jnp.arange(y - w_y + 1)
    startsz = jnp.arange(z - w_z + 1)
    starts_xyz = jnp.stack(
        jnp.meshgrid(startsx, startsy, startsz, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    return jax.vmap(
        lambda start: jax.lax.dynamic_slice(
            matrix, (start[0], start[1], start[2]), (w_x, w_y, w_z)
        )
    )(starts_xyz)


def glcm_kernel(base, neighbour, levels):
    n = base.shape[0]
    level = jnp.arange(n)
    result = jnp.zeros((n, levels, levels))
    result = result.at[level, base, neighbour].set(1)
    result_1 = result.at[level, neighbour, base].set(1)
    diagonal = jnp.eye(levels) + 1
    result_normed = jnp.sum(result_1 * diagonal, axis=0) / (2 * n)

    return result_normed


def glcm_compute_D0(window, levels):  # EAST
    base = window[:, :, :-1].flatten()
    neighbour = window[:, :, 1:].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_compute_D1(window, levels=16):  # SOUTH
    base = window[:, :-1, :].flatten()
    neighbour = window[:, 1:, :].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_compute_D2(window, levels):  # SOUTH_EAST
    base = window[:, :-1, :-1].flatten()
    neighbour = window[:, 1:, 1:].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_compute_D3(window, levels):  # SOUTH_WEST
    base = window[:, 1:, -1].flatten()
    neighbour = window[:, :-1, 1:].flatten()
    return glcm_kernel(base, neighbour, levels)


def glcm_base(X, levels=16, direction=1, window_size=7):
    X_min = jnp.min(X)
    X_max = jnp.max(X)
    hw = window_size // 2
    gray_level = ((X - X_min) / (X_max - X_min)) * (levels - 1)
    gray_level = gray_level.astype("int32")
    gl_pad = jnp.pad(
        gray_level,
        pad_width=((0, 0), (hw, hw), (hw, hw)),
        mode="constant",
        constant_values=0,
    )
    windows = moving_window(gl_pad, (1, window_size, window_size))
    if direction == 0:
        f = jax.vmap(glcm_compute_D0, in_axes=0)
    elif direction == 1:
        f = jax.vmap(glcm_compute_D1, in_axes=0)
    elif direction == 2:
        f = jax.vmap(glcm_compute_D2, in_axes=0)
    elif direction == 3:
        f = jax.vmap(glcm_compute_D3, in_axes=0)
    n_windows = windows.shape[0]
    windows_split = jnp.array_split(windows, n_windows // 4096)
    windows_split = jnp.array(windows_split)
    return jax.lax.map(f, windows_split).reshape(*X.shape, levels, levels)


def glcm_dissimilarity(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I, J = jnp.ogrid[0:levels, 0:levels]
    weights = jnp.abs(I - J)
    return jnp.sum(weights * glcm, axis=(-2, -1))


def glcm_homogeneity(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I, J = jnp.ogrid[0:levels, 0:levels]
    weights = 1.0 / (1.0 + (I - J) ** 2)
    return jnp.sum(weights * glcm, axis=(-2, -1))


def glcm_contrast(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I, J = jnp.ogrid[0:levels, 0:levels]
    weights = (I - J) ** 2
    return jnp.sum(weights * glcm, axis=(-2, -1))


def glcm_energy(X, levels=16, direction=1):
    results = jnp.sqrt(glcm_asm(X, levels, direction))
    return results


def glcm_asm(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    results = jnp.sum(glcm**2, axis=(-2, -1))
    return results


def glcm_correlation(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I, J = jnp.ogrid[0:levels, 0:levels]
    return


def glcm_mean(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I = jnp.np.arange(levels)
    mean = jnp.sum(I * glcm, axis=(-2, -1))
    return mean


def glcm_variance(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    I = jnp.np.arange(levels)
    mean = jnp.sum(I * glcm, axis=(-2, -1))
    return jnp.sum(glcm * ((I - mean) ** 2), axis=(-2, -1))


def glcm_std(X, levels=16, direction=1):
    variance = glcm_variance(X, levels, direction)
    return jnp.sqrt(variance)


def glcm_entropy(X, levels=16, direction=1):
    glcm = glcm_base(X, levels, direction)
    glcm_wo_0s = glcm.at(jnp.where(X == 0)).set(1)
    log = -jnp.log(glcm_wo_0s)
    results = np.sum(glcm * log, axis=(-2, -1))
    return results
