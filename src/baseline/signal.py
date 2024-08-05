try:
    import cupyx.scipy.signal as cusignal
    import cupy.fft as cufft
except:
    pass
import scipy.signal as signal
import scipy.fft as fft

from dasf.transforms import Transform


class FFT(Transform):
    def _transform(self, X, xp):
        return xp.fft(X, axis=-1)

    def _transform_cpu(self, X):
        return self._transform(X, fft)

    def _transform_gpu(self, X):
        return self._transform(X, cufft)


class Convolve1D(Transform):
    def _transform(self, X, w, xp):
        return xp.convolve(X, w, mode="same", method="direct")

    def _transform_cpu(self, X, w):
        return self._transform(X, w, signal)

    def _transform_gpu(self, X, w):
        return self._transform(X, w, cusignal)


class Convolve2D(Transform):
    def _transform(self, X, w, xp):
        return xp.convolve2d(X, w, mode="same")

    def _transform_cpu(self, X, w):
        return self._transform(X, w, signal)

    def _transform_gpu(self, X, w):
        return self._transform(X, w, cusignal)


class Convolve3D(Transform):
    def _transform(self, X, w, xp):
        return xp.convolve(X, w, mode="same", method="direct")

    def _transform_cpu(self, X, w):
        return self._transform(X, w, signal)

    def _transform_gpu(self, X, w):
        return self._transform(X, w, cusignal)


class Correlate1D(Transform):
    def _transform(self, X, w, xp):
        return xp.correlate(X, w, mode="same", method="direct")

    def _transform_cpu(self, X, w):
        return self._transform(X, w, signal)

    def _transform_gpu(self, X, w):
        return self._transform(X, w, cusignal)


class Correlate2D(Transform):
    def _transform(self, X, w, xp):
        return xp.correlate2d(X, w, mode="same")

    def _transform_cpu(self, X, w):
        return self._transform(X, w, signal)

    def _transform_gpu(self, X, w):
        return self._transform(X, w, cusignal)


class Correlate3D(Transform):
    def _transform(self, X, w, xp):
        return xp.correlate(X, w, mode="same", method="direct")

    def _transform_cpu(self, X, w):
        return self._transform(X, w, signal)

    def _transform_gpu(self, X, w):
        return self._transform(X, w, cusignal)
