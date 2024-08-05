import jax.numpy as jnp
import numpy as np
from .utils import first_derivative


def fft(X):
    return jnp.fft.fft(X, axis=-1)


def hilbert(X):
    _, _, z = X.shape
    freq = jnp.fft.fft(X, axis=-1)
    h = jnp.zeros((z,), dtype=X.dtype)
    if z % 2 == 0:
        h = h.at[0].set(1)
        h = h.at[z // 2].set(1)
        h = h.at[1 : z // 2].set(2)
    else:
        h = h.at[0].set(1)
        h = h.at[1 : (z + 1) // 2].set(2)
    h = jnp.reshape(h, (1, 1, z))
    freq = freq * h
    imag = jnp.imag(jnp.fft.ifft(freq, axis=-1))
    return X + 1j * imag


def envelope(X):
    return jnp.abs(hilbert(X))


def instantaneous_phase(X):
    return jnp.angle(hilbert(X), deg=True)


def cosine_instantaneous_phase(X):
    return jnp.cos(jnp.angle(hilbert(X), deg=False))


def relative_amplitude_change(X):
    env = envelope(X)
    env_prime = first_derivative(env)

    return jnp.clip(env_prime / env, -1, 1)


def amplitude_acceleration(X):
    rac = relative_amplitude_change(X)
    return first_derivative(rac)


def instantaneous_frequency(X, sample_rate=4):
    fs = 1000 / sample_rate
    phase = jnp.angle(hilbert(X), deg=False)
    phase = jnp.unwrap(phase)
    phase_prime = first_derivative(phase)
    return jnp.abs((phase_prime / (2.0 * np.pi)) * fs)


def instantaneous_bandwidth(X):
    rac = relative_amplitude_change(X)
    return jnp.abs(rac) / (2.0 * np.pi)


def dominant_frequency(X, sample_rate=4):
    inst_freq = instantaneous_frequency(X, sample_rate=sample_rate)
    inst_band = instantaneous_bandwidth(X)
    return jnp.hypot(inst_freq, inst_band)


def frequency_change(X, sample_rate=4):
    inst_freq = instantaneous_frequency(X, sample_rate=sample_rate)
    return first_derivative(inst_freq)


def sweetness(X, sample_rate=4):
    inst_freq = instantaneous_frequency(X, sample_rate=sample_rate)
    inst_freq = jnp.where(inst_freq < 5, 5, inst_freq)
    env = envelope(X)
    return env / inst_freq


def quality_factor(X, sample_rate=4):
    inst_freq = instantaneous_frequency(X, sample_rate=sample_rate)
    rac = relative_amplitude_change(X)
    return (np.pi * inst_freq) / rac


def response_phase(X):
    env = envelope(X)
    phase = instantaneous_phase(X)
    troughs = local_troughs(env)
    troughs = jnp.cumsum(troughs, axis=-1)
    return response_operation(env, phase, troughs)


def response_frequency(X, sample_rate=4):
    env = envelope(X)
    inst_freq = instantaneous_frequency(X, sample_rate=sample_rate)
    troughs = local_troughs(env)
    troughs = jnp.cumsum(troughs, axis=-1)
    return response_operation(env, inst_freq, troughs)


def response_amplitude(X):
    env = envelope(X)
    troughs = local_troughs(env)
    troughs = jnp.cumsum(troughs, axis=-1)
    return response_operation(env, X, troughs)


def apparent_polarity(X):
    env = envelope(X)
    troughs = local_troughs(env)
    troughs = jnp.cumsum(troughs, axis=-1)
    return polarity_operation(env, X, troughs)


# Auxiliary Functions
def local_troughs(X):
    _, _, z = X.shape
    # Config take indexes
    idx = jnp.arange(0, z)
    idx_plus = idx + 1
    idx_plus = idx_plus.at[z - 1].set(z - 1)
    idx_minus = idx - 1
    idx_minus = idx_minus.at[0].set(0)

    # Get trace and neighbours
    trace = jnp.take(X, idx, axis=-1, unique_indices=True, indices_are_sorted=True)
    plus = jnp.take(X, idx_plus, axis=-1, indices_are_sorted=True)
    minus = jnp.take(X, idx_minus, axis=-1, indices_are_sorted=True)

    # Find troughts
    result = jnp.ones(X.shape, dtype=bool)
    result = jnp.logical_and(result, jnp.less(trace, plus))
    result = jnp.logical_and(result, jnp.less(trace, minus))
    return result


def local_peaks(X):
    _, _, z = X.shape
    # Config take indexes
    idx = jnp.arange(0, z)
    idx_plus = idx + 1
    idx_plus = idx_plus.at[z - 1].set(z - 1)
    idx_minus = idx - 1
    idx_minus = idx_minus.at[0].set(0)

    # Get trace and neighbours
    trace = jnp.take(X, idx, axis=-1, unique_indices=True, indices_are_sorted=True)
    plus = jnp.take(X, idx_plus, axis=-1, indices_are_sorted=True)
    minus = jnp.take(X, idx_minus, axis=-1, indices_are_sorted=True)

    # Find peaks
    result = jnp.ones(X.shape, dtype=bool)
    result = jnp.logical_and(result, jnp.greater(trace, plus))
    result = jnp.logical_and(result, jnp.greater(trace, minus))
    return result


def response_operation(X1, X2, X3):
    x, y, z = X1.shape
    out = jnp.zeros((x, y, z), dtype=X2.dtype)
    all_idx = jnp.arange(0, z)
    for i, j in np.ndindex(out.shape[:-1]):
        ints = jnp.unique(X3[i, j, :], size=z, fill_value=-1)
        for ii in range(z):
            idx = all_idx[jnp.where(X3[i, j, :] == ints[ii], True, False)]
            peak = idx[X1[i, j, idx].argmax()]
            out = out.at[i, j, idx].set(X2[i, j, peak])

    return out


def polarity_operation(X1, X2, X3):
    x, y, z = X1.shape
    out = jnp.zeros((x, y, z), dtype=X2.dtype)
    all_idx = jnp.arange(0, z)
    for i, j in np.ndindex(out.shape[:-1]):
        ints = jnp.unique(X3[i, j, :], size=z, fill_value=-1)
        for ii in range(z):
            if ints[ii] >= 0:
                idx = all_idx[jnp.where(X3[i, j, :] == ints[ii], True, False)]
                peak = idx[X1[i, j, idx].argmax()]
                out = out.at[i, j, idx].set(X1[i, j, peak] * jnp.sign(X2[i, j, peak]))
            else:
                break

    return out
