from tvm import te, tir
import tvm.topi as topi
import numpy as np

from tvm_te_operators.complex_trace.fft import FFT, IFFT
from tvm_te_operators.utils import (
    get_name,
)


class GLCMBase:
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        self._glcm_size = glcm_size
        self._window_size = window_size
        self._direction = direction
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        result = self._computation_kernel(
            X,
            x,
            y,
            z,
        )

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        mi = 0
        ma = 10
        glcm_size = te.const(self._glcm_size, "int")
        window_size = te.const(self._window_size, "int")
        pad = self._window_size // 2

        Xpad = te.compute(
            (x, y + 2 * pad, z + 2 * pad),
            lambda i, j, k: te.if_then_else(
                te.all(j >= pad, j - pad < y, k >= pad, k - pad < z),
                X[i, j - pad, k - pad],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )

        gray_scale = te.compute(
            (x, y + 2 * pad, z + 2 * pad),
            lambda i, j, k: te.trunc(((Xpad[i, j, k] - mi) / (ma - mi)) * glcm_size),
            name=get_name("gray_scale"),
        )

        if self._direction == 0:  # EAST
            glcm_exp = te.compute(
                (x, y, z, glcm_size, glcm_size, window_size, window_size - 1),
                lambda i, j, k, idx, idy, w1, w2: te.if_then_else(
                    te.all(
                        gray_scale[i, j + w1, k + w2] == idx,
                        gray_scale[i, j + w1, k + w2 + 1] == idy,
                    ),
                    te.const(1, "int"),
                    te.const(0, "int"),
                ),
                name=get_name("glcm_exp"),
            )
            r1 = te.reduce_axis((0, window_size), name="r1")
            r2 = te.reduce_axis((0, window_size - 1), name="r2")
            total = te.const(self._window_size * (self._window_size - 1), "float")
        elif self._direction == 1:  # SOUTH
            glcm_exp = te.compute(
                (x, y, z, glcm_size, glcm_size, window_size - 1, window_size),
                lambda i, j, k, idx, idy, w1, w2: te.if_then_else(
                    te.all(
                        gray_scale[i, j + w1, k + w2] == idx,
                        gray_scale[i, j + w1 + 1, k + w2] == idy,
                    ),
                    te.const(1, "int"),
                    te.const(0, "int"),
                ),
                name=get_name("glcm_exp"),
            )
            r1 = te.reduce_axis((0, window_size - 1), name="r1")
            r2 = te.reduce_axis((0, window_size), name="r2")
            total = te.const(self._window_size * (self._window_size - 1), "float")
        elif self._direction == 2:  # SOTH_EAST
            glcm_exp = te.compute(
                (x, y, z, glcm_size, glcm_size, window_size - 1, window_size - 1),
                lambda i, j, k, idx, idy, w1, w2: te.if_then_else(
                    te.all(
                        gray_scale[i, j + w1, k + w2] == idx,
                        gray_scale[i, j + w1 + 1, k + w2 + 1] == idy,
                    ),
                    te.const(1, "int"),
                    te.const(0, "int"),
                ),
                name=get_name("glcm_exp"),
            )
            r1 = te.reduce_axis((0, window_size - 1), name="r1")
            r2 = te.reduce_axis((0, window_size - 1), name="r2")
            total = te.const((self._window_size - 1) ** 2, "float")
        elif self._direction == 3:  # SOTH_WEST
            glcm_exp = te.compute(
                (x, y, z, glcm_size, glcm_size, window_size - 1, window_size - 1),
                lambda i, j, k, idx, idy, w1, w2: te.if_then_else(
                    te.all(
                        gray_scale[i, j + w1 + 1, k + w2] == idx,
                        gray_scale[i, j + w1, k + w2 + 1] == idy,
                    ),
                    te.const(1, "int"),
                    te.const(0, "int"),
                ),
                name=get_name("glcm_exp"),
            )
            r1 = te.reduce_axis((0, window_size - 1), name="r1")
            r2 = te.reduce_axis((0, window_size - 1), name="r2")
            total = te.const((self._window_size - 1) ** 2, "float")

        glcm = te.compute(
            (x, y, z, glcm_size, glcm_size),
            lambda i, j, k, idx, idy: te.div(
                te.sum(glcm_exp[i, j, k, idx, idy], axis=[r1, r2]), total
            ),
            name=get_name("glcm"),
        )

        return glcm


class GLCMEntropy(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")
        log = te.compute(
            (x, y, z, self._glcm_size, self._glcm_size),
            lambda i, j, k, idx, idy: te.if_then_else(
                X[i, j, k] == 0,
                te.const(0.0, "float32") - tir.log(X[i, j, k, idx, idy]),
            ),
            name=get_name("log"),
        )

        entropy = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                X[i, j, k, r1, r2] * log[i, j, k, r1, r2], axis=[r1, r2]
            ),
            name=get_name("entropy"),
        )

        return entropy


class GLCMContrast(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        contrast = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                X[i, j, k, r1, r2] * te.power(r1 - r2, 2), axis=[r1, r2]
            ),
            name=get_name("contrast"),
        )
        return contrast


class GLCMDissimilarity(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        dissimilarity = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(X[i, j, k, r1, r2] * te.abs(r1 - r2), axis=[r1, r2]),
            name=get_name("dissimilarity"),
        )
        return dissimilarity


class GLCMHomogeneity(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        homogeneity = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                te.div(X[i, j, k, r1, r2], 1 + te.power(r1 - r2, 2)), axis=[r1, r2]
            ),
            name=get_name("homogeneity"),
        )
        return homogeneity


class GLCMASM(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        asm = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(te.power(X[i, j, k, r1, r2], 2), axis=[r1, r2]),
            name=get_name("asm"),
        )
        return asm


class GLCMEnergy(GLCMASM):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        asm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(asm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        energy = te.compute(
            (x, y, z),
            lambda i, j, k: te.sqrt(X[i, j, k]),
            name=get_name("energy"),
        )
        return energy


class GLCMMean(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        mean = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(r1 * X[i, j, k, r1, r2], axis=[r1, r2]),
            name=get_name("mean"),
        )
        return mean


class GLCMVariance(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        mean = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(r1 * X[i, j, k, r1, r2], axis=[r1, r2]),
            name=get_name("mean"),
        )

        variance = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                X[i, j, k, r1, r2] * te.power(r1 - mean, 2), axis=[r1, r2]
            ),
            name=get_name("variance"),
        )
        return variance


class GLCMStandardDeviation(GLCMVariance):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        variance = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(variance, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        std = te.compute(
            (x, y, z),
            lambda i, j, k: te.sqrt(X[i, j, k]),
            name=get_name("std"),
        )
        return std


class GLCMCorrelation(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=0, computation_context={}
    ):
        super().__init__(
            glcm_size=glcm_size,
            window_size=window_size,
            direction=direction,
            computation_context=computation_context,
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )

        result = self._computation_kernel(glcm, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        mean = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(r1 * X[i, j, k, r1, r2], axis=[r1, r2]),
            name=get_name("mean"),
        )

        variance = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                X[i, j, k, r1, r2] * te.power(r1 - mean, 2), axis=[r1, r2]
            ),
            name=get_name("variance"),
        )

        correlation = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                te.div((r1 - mean) * (r2 - mean), variance[i, j, k]), axis=[r1, r2]
            ),
            name=get_name("correlation"),
        )
        return correlation
