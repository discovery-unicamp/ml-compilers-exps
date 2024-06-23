from tvm import te, tir
import tvm.topi as topi
import numpy as np

from tvm_te_operators.utils import (
    get_name,
)


class GLCMBase:
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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
        rx = te.reduce_axis((0, x), name="rx")
        ry = te.reduce_axis((0, y), name="ry")
        rz = te.reduce_axis((0, z), name="rz")

        # Xmi = te.compute((x,y,), lambda i, j: te.min(X[i, j, rz], axis=rz), name="Xmi_z")
        # Xmi = te.compute((x,), lambda i: te.min(Xmi[i, ry], axis=ry), name="Xmi_y")
        Xmi = te.compute(
            (1,), lambda _: te.min(X[rx, ry, rz], axis=[rx, ry, rz]), name="Xmi"
        )

        rx = te.reduce_axis((0, x), name="rx")
        ry = te.reduce_axis((0, y), name="ry")
        rz = te.reduce_axis((0, z), name="rz")
        # Xma = te.compute((x,y,), lambda i, j: te.max(X[i, j, rz], axis=rz), name="Xma_z")
        # Xma = te.compute((x,), lambda i: te.max(Xma[i, ry], axis=ry), name="Xma_y")
        Xma = te.compute(
            (1,), lambda _: te.max(X[rx, ry, rz], axis=[rx, ry, rz]), name="Xma"
        )

        glcm_size = te.const(self._glcm_size, "int")
        window_size = te.const(self._window_size, "int")
        pad = self._window_size // 2

        gray_scale = te.compute(
            (x, y, z),
            lambda i, j, k: te.floor(
                ((X[i, j, k] - Xmi[0]) / (Xma[0] - Xmi[0])) * (glcm_size - 1)
            ),
            name=get_name("gray_scale"),
        )

        gray_scale_padded = te.compute(
            (x, y + 2 * pad, z + 2 * pad),
            lambda i, j, k: te.if_then_else(
                te.all(j >= pad, j - pad < y, k >= pad, k - pad < z),
                gray_scale[i, j - pad, k - pad],
                te.const(0.0, X.dtype),
            ),
            name=get_name("gray_scale_padded"),
        )

        glcm_windows = te.compute(
            (x, y, z, window_size, window_size),
            lambda i, j, k, w1, w2: gray_scale_padded[i, j + w1, k + w2],
            name=get_name("glcm_exp"),
        )

        if self._direction == 0:  # EAST
            r1 = te.reduce_axis((0, window_size), name="r1")
            r2 = te.reduce_axis((0, window_size - 1), name="r2")
            glcm = te.compute(
                (x, y, z, glcm_size, glcm_size),
                lambda i, j, k, idx, idy: te.sum(
                    te.if_then_else(
                        te.any(
                            te.all(
                                glcm_windows[i, j, k, r1, r2] == idx,
                                glcm_windows[i, j, k, r1, r2 + 1] == idy,
                            ),
                            te.all(
                                glcm_windows[i, j, k, r1, r2] == idy,
                                glcm_windows[i, j, k, r1, r2 + 1] == idx,
                            ),
                        ),
                        te.if_then_else(
                            idx == idy,
                            te.const(2, "int"),
                            te.const(1, "int"),
                        ),
                        te.const(0, "int"),
                    ),
                    axis=[r1, r2],
                ),
                name=get_name("glcm"),
            )
            total = te.const(2 * self._window_size * (self._window_size - 1), X.dtype)
        elif self._direction == 1:  # SOUTH
            r1 = te.reduce_axis((0, window_size - 1), name="r1")
            r2 = te.reduce_axis((0, window_size), name="r2")

            glcm = te.compute(
                (x, y, z, glcm_size, glcm_size),
                lambda i, j, k, idx, idy: te.sum(
                    te.if_then_else(
                        te.any(
                            te.all(
                                glcm_windows[i, j, k, r1, r2] == idx,
                                glcm_windows[i, j, k, r1 + 1, r2] == idy,
                            ),
                            te.all(
                                glcm_windows[i, j, k, r1, r2] == idy,
                                glcm_windows[i, j, k, r1 + 1, r2] == idx,
                            ),
                        ),
                        te.if_then_else(
                            idx == idy,
                            te.const(2, "int"),
                            te.const(1, "int"),
                        ),
                        te.const(0, "int"),
                    ),
                    axis=[r1, r2],
                ),
                name=get_name("glcm"),
            )
            total = te.const(2 * self._window_size * (self._window_size - 1), X.dtype)
        elif self._direction == 2:  # SOTH_EAST
            r1 = te.reduce_axis((0, window_size - 1), name="r1")
            r2 = te.reduce_axis((0, window_size - 1), name="r2")
            glcm = te.compute(
                (x, y, z, glcm_size, glcm_size),
                lambda i, j, k, idx, idy: te.sum(
                    te.if_then_else(
                        te.any(
                            te.all(
                                glcm_windows[i, j, k, r1, r2] == idx,
                                glcm_windows[i, j, k, r1 + 1, r2 + 1] == idy,
                            ),
                            te.all(
                                glcm_windows[i, j, k, r1, r2] == idy,
                                glcm_windows[i, j, k, r1 + 1, r2 + 1] == idx,
                            ),
                        ),
                        te.if_then_else(
                            idx == idy,
                            te.const(2, "int"),
                            te.const(1, "int"),
                        ),
                        te.const(0, "int"),
                    ),
                    axis=[r1, r2],
                ),
                name=get_name("glcm"),
            )
            total = te.const(2 * (self._window_size - 1) ** 2, X.dtype)
        elif self._direction == 3:  # SOTH_WEST
            r1 = te.reduce_axis((0, window_size - 1), name="r1")
            r2 = te.reduce_axis((0, window_size - 1), name="r2")

            glcm = te.compute(
                (x, y, z, glcm_size, glcm_size),
                lambda i, j, k, idx, idy: te.sum(
                    te.if_then_else(
                        te.any(
                            te.all(
                                glcm_windows[i, j, k, r1 + 1, r2] == idx,
                                glcm_windows[i, j, k, r1, r2 + 1] == idy,
                            ),
                            te.all(
                                glcm_windows[i, j, k, r1 + 1, r2] == idy,
                                glcm_windows[i, j, k, r1, r2 + 1] == idx,
                            ),
                        ),
                        te.if_then_else(
                            idx == idy,
                            te.const(2, "int"),
                            te.const(1, "int"),
                        ),
                        te.const(0, "int"),
                    ),
                    axis=[r1, r2],
                ),
                name=get_name("glcm"),
            )
            total = te.const(2 * (self._window_size - 1) ** 2, X.dtype)

        glcm_normed = te.compute(
            (x, y, z, glcm_size, glcm_size),
            lambda i, j, k, idx, idy: te.div(glcm[i, j, k, idx, idy], total),
            name=get_name("glcm_normed"),
        )

        return (glcm_normed,)


class GLCMEntropy(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]

        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")
        log = te.compute(
            (x, y, z, self._glcm_size, self._glcm_size),
            lambda i, j, k, idx, idy: te.if_then_else(
                glcm[i, j, k, idx, idy] == 0,
                te.const(0.0, "float32"),
                -tir.log(glcm[i, j, k, idx, idy]),
            ),
            name=get_name("log"),
        )

        entropy = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                glcm[i, j, k, r1, r2] * log[i, j, k, r1, r2], axis=[r1, r2]
            ),
            name=get_name("entropy"),
        )

        return (entropy,)


class GLCMContrast(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        contrast = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                glcm[i, j, k, r1, r2] * te.power(tir.Cast("float", r1 - r2), 2),
                axis=[r1, r2],
            ),
            name=get_name("contrast"),
        )
        return (contrast,)


class GLCMDissimilarity(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        dissimilarity = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                glcm[i, j, k, r1, r2] * te.abs(r1 - r2), axis=[r1, r2]
            ),
            name=get_name("dissimilarity"),
        )
        return (dissimilarity,)


class GLCMHomogeneity(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        homogeneity = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                te.div(
                    glcm[i, j, k, r1, r2], 1 + te.power(tir.Cast("float", r1 - r2), 2)
                ),
                axis=[r1, r2],
            ),
            name=get_name("homogeneity"),
        )
        return (homogeneity,)


class GLCMASM(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        asm = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(te.power(glcm[i, j, k, r1, r2], 2), axis=[r1, r2]),
            name=get_name("asm"),
        )
        return (asm,)


class GLCMEnergy(GLCMASM):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        asm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]
        energy = te.compute(
            (x, y, z),
            lambda i, j, k: te.sqrt(asm[i, j, k]),
            name=get_name("energy"),
        )
        return (energy,)


class GLCMMean(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        mean = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(r2 * glcm[i, j, k, r1, r2], axis=[r1, r2]),
            name=get_name("mean"),
        )
        return (mean,)


class GLCMVariance(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        mean = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(r1 * glcm[i, j, k, r1, r2], axis=[r1, r2]),
            name=get_name("mean"),
        )
        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")
        variance = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                glcm[i, j, k, r1, r2] * te.power(r1 - mean[i, j, k], 2), axis=[r1, r2]
            ),
            name=get_name("variance"),
        )
        return (variance,)


class GLCMStandardDeviation(GLCMVariance):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        variance = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]

        std = te.compute(
            (x, y, z),
            lambda i, j, k: te.sqrt(variance[i, j, k]),
            name=get_name("std"),
        )
        return (std,)


class GLCMCorrelation(GLCMBase):
    def __init__(
        self, glcm_size=16, window_size=7, direction=1, computation_context={}
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

        result = self._computation_kernel(X, x, y, z)

        self.computation_context = {
            "result": result,
            "input": (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z):
        glcm = super()._computation_kernel(
            X,
            x,
            y,
            z,
        )[0]

        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        mean = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(r1 * glcm[i, j, k, r1, r2], axis=[r1, r2]),
            name=get_name("mean"),
        )

        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        variance = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                glcm[i, j, k, r1, r2] * te.power(r1 - mean[i, j, k], 2), axis=[r1, r2]
            ),
            name=get_name("variance"),
        )

        variance_wo_0s = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                variance[i, j, k] == 0, te.const(1, "float32"), variance[i, j, k]
            ),
            name=get_name("variance_wo_0s"),
        )

        r1 = te.reduce_axis((0, self._glcm_size), name="r1")
        r2 = te.reduce_axis((0, self._glcm_size), name="r2")

        correlation = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                glcm[i, j, k, r1, r2]
                * te.div(
                    (r1 - mean[i, j, k]) * (r2 - mean[i, j, k]), variance_wo_0s[i, j, k]
                ),
                axis=[r1, r2],
            ),
            name=get_name("correlation"),
        )
        return (correlation,)
