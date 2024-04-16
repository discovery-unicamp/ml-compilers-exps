import numpy as np
import cupy as cp
import cupyx.scipy.signal as cusignal
import scipy.signal as signal

from dasf.transforms import Transform
weights_1d_np = np.array([-0.5, 0.0, 0.5])
weights_2d_np = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1 , -1]
])

weights_1d_cp = cp.array([-0.5, 0.0, 0.5])
weights_2d_cp = cp.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1 , -1]
])


class Convolve1D(Transform):
    def _transform(self, X, w, xp):
        return xp.convolve(X, w, mode="same", method="direct")

    def _transform_cpu(self, X):
        return self._transform(X, weights_1d_np, signal)
    
    def _transform_gpu(self, X, **kwargs):
        return self._transform(X, weights_1d_cp, cusignal)



class Convolve2D(Transform):
    def _transform(self, X, w, xp):
        return xp.convolve2d(X, w, mode="same")

    def _transform_cpu(self, X):
        return self._transform(X, weights_2d_np, signal)
    
    def _transform_gpu(self, X, **kwargs):
        return self._transform(X, weights_2d_cp, cusignal)
    

class Correlate1D(Transform):
    def _transform(self, X, w, xp):
        return xp.correlate(X, w, mode="same", method="direct")

    def _transform_cpu(self, X):
        return self._transform(X, weights_1d_np, signal)
    
    def _transform_gpu(self, X, **kwargs):
        return self._transform(X, weights_1d_cp, cusignal)


class Correlate2D(Transform):
    def _transform(self, X, w, xp):
        return xp.correlate2d(X, w, mode="same")

    def _transform_cpu(self, X):
        return self._transform(X, weights_2d_np, signal)
    
    def _transform_gpu(self, X, **kwargs):
        return self._transform(X, weights_2d_cp, cusignal)

