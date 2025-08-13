try:
    import cupyx.scipy.signal as cusignal
    import cupy.fft as cufft
    import cupy as cp
except:
    pass
import scipy.signal as signal
import scipy.fft as fft


class FFT:
    @staticmethod
    def _transform_cpu(X):
        return fft.fft(X, axis=-1)

    @staticmethod
    def _transform_gpu(X):
        ret = cufft.fft(X, axis=-1)
        cp.cuda.runtime.deviceSynchronize()
        return ret


class Convolve1D:
    @staticmethod
    def _transform_cpu(X, w):
        return signal.convolve(X, w, mode="same", method="direct")

    @staticmethod
    def _transform_gpu(X, w):
        return cusignal.convolve(X, w, mode="same", method="direct")


class Convolve2D:
    @staticmethod
    def _transform_cpu(X, w):
        return signal.convolve2d(X, w, mode="same")

    @staticmethod
    def _transform_gpu(X, w):
        return cusignal.convolve2d(X, w, mode="same")


class Convolve3D:
    @staticmethod
    def _transform_cpu(X, w):
        return signal.convolve(X, w, mode="same", method="direct")

    @staticmethod
    def _transform_gpu(X, w):
        return cusignal.convolve(X, w, mode="same", method="direct")


class Correlate1D:
    @staticmethod
    def _transform_cpu(X, w):
        return signal.correlate(X, w, mode="same", method="direct")

    @staticmethod
    def _transform_gpu(X, w):
        return cusignal.correlate(X, w, mode="same", method="direct")


class Correlate2D:
    @staticmethod
    def _transform_cpu(X, w):
        return signal.correlate2d(X, w, mode="same")

    @staticmethod
    def _transform_gpu(X, w):
        return cusignal.correlate2d(X, w, mode="same")


class Correlate3D:
    @staticmethod
    def _transform_cpu(X, w):
        return signal.correlate(X, w, mode="same", method="direct")

    @staticmethod
    def _transform_gpu(X, w):
        return cusignal.correlate(X, w, mode="same", method="direct")
