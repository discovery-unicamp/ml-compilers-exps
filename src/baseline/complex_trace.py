# Implementation based on https://github.com/dudley-fitzgerald/d2geo/blob/master/attributes/CompleTrace.py and https://github.com/dudley-fitzgerald/d2geo/blob/master/attributes/SignalProcess.py
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndi
from numba import cuda, njit

try:
    import cupy as cp
    import cupyx.scipy.signal as cusignal
    from cupyx.scipy import ndimage as cundi
except ImportError:
    pass


class FirstDerivative:
    @staticmethod
    def _transform_cpu(X, axis=-1):
        axes = [ax for ax in range(X.ndim) if ax != axis]

        result0 = ndi.correlate1d(X, weights=np.array([-0.5, 0.0, 0.5]), axis=axis)

        result1 = ndi.correlate1d(
            result0, weights=np.array([0.178947, 0.642105, 0.178947]), axis=axes[0]
        )

        result2 = ndi.correlate1d(
            result1, weights=np.array([0.178947, 0.642105, 0.178947]), axis=axes[1]
        )

        return result2

    @staticmethod
    def _transform_gpu(X, axis=-1):
        axes = [ax for ax in range(X.ndim) if ax != axis]

        result0 = cundi.correlate1d(X, weights=cp.array([-0.5, 0.0, 0.5]), axis=axis)

        result1 = cundi.correlate1d(
            result0, weights=cp.array([0.178947, 0.642105, 0.178947]), axis=axes[0]
        )

        result2 = cundi.correlate1d(
            result1, weights=cp.array([0.178947, 0.642105, 0.178947]), axis=axes[1]
        )

        return result2


class Hilbert:
    @staticmethod
    def _transform_cpu(X):
        return signal.hilbert(X, axis=-1)

    @staticmethod
    def _transform_gpu(X):
        return cusignal.hilbert(X, axis=-1)


class Envelope:
    @staticmethod
    def _transform_cpu(X):
        return np.absolute(Hilbert._transform_cpu(X))

    @staticmethod
    def _transform_gpu(X):
        return cp.absolute(Hilbert._transform_gpu(X))


class InstantaneousPhase:
    @staticmethod
    def _transform_cpu(X):
        return np.rad2deg(np.angle(Hilbert._transform_cpu(X)))

    @staticmethod
    def _transform_gpu(X):
        return cp.rad2deg(cp.angle(Hilbert._transform_gpu(X)))


class CosineInstantaneousPhase(Hilbert):
    @staticmethod
    def _transform_cpu(X):
        return np.cos(np.angle(Hilbert._transform_cpu(X)))

    @staticmethod
    def _transform_gpu(X):
        return cp.cos(cp.angle(Hilbert._transform_gpu(X)))


class RelativeAmplitudeChange:
    @staticmethod
    def _transform_cpu(X):
        env = Envelope._transform_cpu(X)
        env_prime = FirstDerivative._transform_cpu(env, axis=-1)

        result = env_prime / env

        result[np.isnan(result)] = 0

        return np.clip(result, -1, 1)

    @staticmethod
    def _transform_gpu(X):
        env = Envelope._transform_gpu(X)
        env_prime = FirstDerivative._transform_gpu(env, axis=-1)

        result = env_prime / env

        result[cp.isnan(result)] = 0

        return cp.clip(result, -1, 1)


class AmplitudeAcceleration:
    @staticmethod
    def _transform_cpu(X):
        rac = RelativeAmplitudeChange._transform_cpu(X)

        return FirstDerivative._transform_cpu(rac, axis=-1)

    @staticmethod
    def _transform_gpu(X):
        rac = RelativeAmplitudeChange._transform_gpu(X)

        return FirstDerivative._transform_gpu(rac, axis=-1)


class InstantaneousFrequency:
    @staticmethod
    def _transform_cpu(X, sample_rate=4):
        fs = 1000 / sample_rate

        phase = InstantaneousPhase._transform_cpu(X)
        phase = np.deg2rad(phase)
        phase = np.unwrap(phase)

        phase_prime = FirstDerivative._transform_cpu(phase, axis=-1)

        return np.absolute((phase_prime / (2.0 * np.pi) * fs))

    @staticmethod
    def _transform_gpu(X, sample_rate=4):
        fs = 1000 / sample_rate

        phase = InstantaneousPhase._transform_gpu(X)
        phase = cp.deg2rad(phase)
        phase = cp.unwrap(phase)

        phase_prime = FirstDerivative._transform_gpu(phase, axis=-1)

        return cp.absolute((phase_prime / (2.0 * np.pi) * fs))


class InstantaneousBandwidth:
    @staticmethod
    def _transform_cpu(X):
        rac = RelativeAmplitudeChange._transform_cpu(X)

        return np.absolute(rac) / (2.0 * np.pi)

    @staticmethod
    def _transform_gpu(X):
        rac = RelativeAmplitudeChange._transform_gpu(X)

        return cp.absolute(rac) / (2.0 * np.pi)


class DominantFrequency:
    @staticmethod
    def _transform_cpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_cpu(X, sample_rate=sample_rate)
        inst_band = InstantaneousBandwidth._transform_cpu(X)
        return np.hypot(inst_freq, inst_band)

    @staticmethod
    def _transform_gpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_gpu(X, sample_rate=sample_rate)
        inst_band = InstantaneousBandwidth._transform_gpu(X)
        return cp.hypot(inst_freq, inst_band)


class FrequencyChange:
    @staticmethod
    def _transform_cpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_cpu(X, sample_rate=sample_rate)
        return FirstDerivative._transform_cpu(inst_freq, axis=-1)

    @staticmethod
    def _transform_gpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_gpu(X, sample_rate=sample_rate)
        return FirstDerivative._transform_gpu(inst_freq, axis=-1)


class Sweetness:
    @staticmethod
    def _transform_cpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_cpu(X, sample_rate=sample_rate)
        inst_freq[inst_freq < 5] = 5
        env = Envelope._transform_cpu(X)

        return env / inst_freq

    @staticmethod
    def _transform_gpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_gpu(X, sample_rate=sample_rate)
        inst_freq[inst_freq < 5] = 5
        env = Envelope._transform_gpu(X)

        return env / inst_freq


class QualityFactor:
    @staticmethod
    def _transform_cpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_cpu(X, sample_rate=sample_rate)
        rac = RelativeAmplitudeChange._transform_cpu(X)
        result = (np.pi * inst_freq) / rac

        result[np.isnan(result)] = 0

        return result

    @staticmethod
    def _transform_gpu(X, sample_rate=4):
        inst_freq = InstantaneousFrequency._transform_gpu(X, sample_rate=sample_rate)
        rac = RelativeAmplitudeChange._transform_gpu(X)
        result = (np.pi * inst_freq) / rac

        result[cp.isnan(result)] = 0

        return result


def local_events(in_data, comparator, is_cupy=False):
    if is_cupy:
        idx = cp.arange(0, in_data.shape[-1])
        trace = in_data.take(idx, axis=-1)
        plus = in_data.take(idx + 1, axis=-1)
        minus = in_data.take(idx - 1, axis=-1)
        plus[:, :, -1] = trace[:, :, -1]
        minus[:, :, 0] = trace[:, :, 0]
        result = cp.ones(in_data.shape, dtype=bool)
    else:
        idx = np.arange(0, in_data.shape[-1])
        trace = in_data.take(idx, axis=-1, mode="clip")
        plus = in_data.take(idx + 1, axis=-1, mode="clip")
        minus = in_data.take(idx - 1, axis=-1, mode="clip")

        result = np.ones(in_data.shape, dtype=bool)

    result &= comparator(trace, plus)
    result &= comparator(trace, minus)

    return result


class ResponsePhase:
    @staticmethod
    def _transform_cpu(X):
        env = Envelope._transform_cpu(X)
        phase = InstantaneousPhase._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_cpu(env, phase, troughs)

        result[np.isnan(result)] = 0

        return result

    @staticmethod
    def _transform_gpu(X):
        env = Envelope._transform_gpu(X)
        phase = InstantaneousPhase._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_gpu(env, phase, troughs)

        result[cp.isnan(result)] = 0

        return result


class ResponseFrequency:
    @staticmethod
    def _transform_cpu(X, sample_rate=4):
        env = Envelope._transform_cpu(X)
        inst_freq = InstantaneousFrequency._transform_cpu(X, sample_rate=sample_rate)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_cpu(env, inst_freq, troughs)

        result[np.isnan(result)] = 0

        return result

    @staticmethod
    def _transform_gpu(X, sample_rate=4):
        env = Envelope._transform_gpu(X)
        inst_freq = InstantaneousFrequency._transform_gpu(X, sample_rate=sample_rate)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_gpu(env, inst_freq, troughs)

        result[cp.isnan(result)] = 0

        return result


class ResponseAmplitude:
    @staticmethod
    def _transform_cpu(X):
        env = Envelope._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_cpu(env, X, troughs)

        result[np.isnan(result)] = 0

        return result

    @staticmethod
    def _transform_gpu(X):
        env = Envelope._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_gpu(env, X, troughs)

        result[cp.isnan(result)] = 0

        return result


class ApparentPolarity:
    @staticmethod
    def _transform_cpu(X):
        env = Envelope._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = polarity_operation_cpu(env, X, troughs)

        result[np.isnan(result)] = 0

        return result

    @staticmethod
    def _transform_gpu(X):
        env = Envelope._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = polarity_operation_gpu(env, X, troughs)

        result[cp.isnan(result)] = 0

        return result


# Numba functions
@njit(parallel=False)
def response_operation_cpu(chunk1, chunk2, chunk3):
    out = np.zeros_like(chunk1)
    for i, j in np.ndindex(out.shape[:-1]):
        ints = np.unique(chunk3[i, j, :])
        for ii in ints:
            idx = np.where(chunk3[i, j, :] == ii)
            idx = idx[0]
            ind = np.zeros(idx.shape[0])
            for k in range(len(idx)):
                ind[k] = chunk1[i, j, idx[k]]
            ind = ind.argmax()
            peak = idx[ind]
            for k in range(len(idx)):
                out[i, j, idx[k]] = chunk2[i, j, peak]
    return out


def response_operation_gpu(chunk1, chunk2, chunk3):
    out = cp.zeros_like(chunk1)
    uniques = cp.zeros_like(chunk1)
    max_ind = cp.zeros_like(chunk1, dtype="int32")
    blockx = out.shape[0]
    blocky = (out.shape[1] + 64 - 1) // 64
    kernel = response_kernel_gpu[(blockx, blocky), (1, 64)]
    kernel(
        chunk1,
        chunk2,
        chunk3,
        out,
        uniques,
        max_ind,
        out.shape[0],
        out.shape[1],
        out.shape[2],
    )
    return out


@cuda.jit()
def response_kernel_gpu(
    chunk1, chunk2, chunk3, out, uniques, max_ind, len_x, len_y, len_z
):
    i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if i < len_x and j < len_y:
        tail_unique = 0
        for k in range(len_z):
            exists = 0
            val = chunk3[i, j, k]
            for l in range(len_z):
                if l < tail_unique:
                    if val == uniques[i, j, l]:
                        exists = 1
                        if chunk1[i, j, k] > chunk1[i, j, max_ind[i, j, l]]:
                            max_ind[i, j, l] = k
            if exists == 0:
                uniques[i, j, tail_unique] = val
                max_ind[i, j, tail_unique] = k
                tail_unique += 1
        for k in range(len_z):
            if k < tail_unique:
                peak_val = chunk2[i, j, max_ind[i, j, k]]
                for l in range(len_z):
                    if chunk3[i, j, l] == uniques[i, j, k]:
                        out[i, j, l] = peak_val


@njit(parallel=False)
def polarity_operation_cpu(chunk1, chunk2, chunk3):
    out = np.zeros_like(chunk1)
    for i, j in np.ndindex(out.shape[:-1]):
        ints = np.unique(chunk3[i, j, :])
        for ii in ints:
            idx = np.where(chunk3[i, j, :] == ii)
            idx = idx[0]
            ind = np.zeros(idx.shape[0])
            for k in range(len(idx)):
                ind[k] = chunk1[i, j, idx[k]]
            ind = ind.argmax()
            peak = idx[ind]
            val = chunk1[i, j, peak] * np.sign(chunk2[i, j, peak])
            for k in range(len(idx)):
                out[i, j, idx[k]] = val
    return out


def polarity_operation_gpu(chunk1, chunk2, chunk3):
    out = cp.zeros_like(chunk1)
    uniques = cp.zeros_like(chunk1)
    max_ind = cp.zeros_like(chunk1, dtype="int32")
    blockx = out.shape[0]
    blocky = (out.shape[1] + 64 - 1) // 64
    kernel = polarity_kernel_gpu[(blockx, blocky), (1, 64)]
    kernel(
        chunk1,
        chunk2,
        chunk3,
        out,
        uniques,
        max_ind,
        out.shape[0],
        out.shape[1],
        out.shape[2],
    )
    return out


@cuda.jit()
def polarity_kernel_gpu(
    chunk1, chunk2, chunk3, out, uniques, max_ind, len_x, len_y, len_z
):
    i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if i < len_x and j < len_y:
        tail_unique = 0
        for k in range(len_z):
            exists = 0
            val = chunk3[i, j, k]
            for l in range(len_z):
                if l < tail_unique:
                    if val == uniques[i, j, l]:
                        exists = 1
                        if chunk1[i, j, k] > chunk1[i, j, max_ind[i, j, l]]:
                            max_ind[i, j, l] = k
            if exists == 0:
                uniques[i, j, tail_unique] = val
                max_ind[i, j, tail_unique] = k
                tail_unique += 1
        for k in range(len_z):
            if k < tail_unique:
                peak_val = chunk1[i, j, max_ind[i, j, k]]
                if chunk2[i, j, max_ind[i, j, k]] < 0:
                    peak_val = -peak_val
                elif chunk2[i, j, max_ind[i, j, k]] == 0:
                    peak_val = 0
                for l in range(len_z):
                    if chunk3[i, j, l] == uniques[i, j, k]:
                        out[i, j, l] = peak_val
