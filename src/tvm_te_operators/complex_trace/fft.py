import tvm
from tvm import te
import tvm.topi as topi
import numpy as np

from tvm_te_operators.utils import get_name


class FFT:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        Y = computation_context.get("Y", None)

        result = self._computation_kernel(
            X,
            Y,
            x,
            y,
            z,
        )

        self.computation_context = {
            "result": result,
            "input": (X, Y) if Y is not None else (X,),
        }
        return self.computation_context

    def _computation_kernel(self, X, Y, x, y, z):
        pi_const = te.const(np.pi, X.dtype)
        stage_real = te.compute(
            (x, y, 2, te.indexdiv(z, 2)),
            lambda i, j, k, l: te.if_then_else(
                k == 0,
                X[i, j, l] + X[i, j, l + te.indexdiv(z, 2)],
                X[i, j, l] - X[i, j, l + te.indexdiv(z, 2)],
            ),
            name=get_name("stage_real_0"),
        )
        if Y is not None:
            stage_imag = te.compute(
                (x, y, 2, te.indexdiv(z, 2)),
                lambda i, j, k, l: te.if_then_else(
                    k == 0,
                    Y[i, j, l] + Y[i, j, l + te.indexdiv(z, 2)],
                    Y[i, j, l] - Y[i, j, l + te.indexdiv(z, 2)],
                ),
                name=get_name("stage_imag_0"),
            )
        else:
            zero = te.const(0, X.dtype)
            stage_imag = te.compute(
                (x, y, 2, te.indexdiv(z, 2)),
                lambda i, j, k, l: zero,
                name=get_name("stage_imag_0"),
            )

        for stage in range(1, int(np.log2(z))):
            hop = 2**stage
            factor_real = te.compute(
                (hop,),
                lambda k: te.cos(te.div((pi_const * k), hop)),
                name=get_name(f"factors_{stage}"),
            )

            factor_imag = te.compute(
                (hop,),
                lambda k: -1 * te.sin(te.div((pi_const * k), hop)),
                name=get_name(f"factors_{stage}"),
            )

            even_real = te.compute(
                (x, y, hop, te.indexdiv(z, 2 * hop)),
                lambda i, j, k, l: stage_real[i, j, k, l],
                name=get_name(f"even_real_{stage}"),
            )

            even_imag = te.compute(
                (x, y, hop, te.indexdiv(z, 2 * hop)),
                lambda i, j, k, l: stage_imag[i, j, k, l],
                name=get_name(f"even_imag_{stage}"),
            )

            odd_real = te.compute(
                (x, y, hop, te.indexdiv(z, 2 * hop)),
                lambda i, j, k, l: stage_real[i, j, k, l + te.indexdiv(z, 2 * hop)]
                * factor_real[k]
                - stage_imag[i, j, k, l + te.indexdiv(z, 2 * hop)] * factor_imag[k],
                name=get_name(f"odd_real_{stage}"),
            )

            odd_imag = te.compute(
                (x, y, hop, te.indexdiv(z, 2 * hop)),
                lambda i, j, k, l: stage_imag[i, j, k, l + te.indexdiv(z, 2 * hop)]
                * factor_real[k]
                + stage_real[i, j, k, l + te.indexdiv(z, 2 * hop)] * factor_imag[k],
                name=get_name(f"odd_imag_{stage}"),
            )
            if te.indexdiv(z, 2 * hop) == 1:
                stage_real = te.compute(
                    (x, y, z),
                    lambda i, j, k: te.if_then_else(
                        k < hop,
                        even_real[i, j, k, 0] + odd_real[i, j, k, 0],
                        even_real[i, j, k - hop, 0] - odd_real[i, j, k - hop, 0],
                    ),
                    name=get_name("fft_real"),
                )
                stage_imag = te.compute(
                    (x, y, z),
                    lambda i, j, k: te.if_then_else(
                        k < hop,
                        even_imag[i, j, k, 0] + odd_imag[i, j, k, 0],
                        even_imag[i, j, k - hop, 0] - odd_imag[i, j, k - hop, 0],
                    ),
                    name=get_name("fft_imag"),
                )
            else:
                stage_real = te.compute(
                    (x, y, 2 * hop, te.indexdiv(z, 2 * hop)),
                    lambda i, j, k, l: te.if_then_else(
                        k < hop,
                        even_real[i, j, k, l] + odd_real[i, j, k, l],
                        even_real[i, j, k - hop, l] - odd_real[i, j, k - hop, l],
                    ),
                    name=get_name(f"stage_real_{stage}"),
                )
                stage_imag = te.compute(
                    (x, y, 2 * hop, te.indexdiv(z, 2 * hop)),
                    lambda i, j, k, l: te.if_then_else(
                        k < hop,
                        even_imag[i, j, k, l] + odd_imag[i, j, k, l],
                        even_imag[i, j, k - hop, l] - odd_imag[i, j, k - hop, l],
                    ),
                    name=get_name(f"stage_imag_{stage}"),
                )

        return stage_real, stage_imag


class IFFT:
    def __init__(self, computation_context={}):
        self._fft = FFT()
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        Y = computation_context.get("Y", te.placeholder((x, y, z), name=get_name("Y")))

        result = self._computation_kernel(
            X,
            Y,
            x,
            y,
            z,
        )

        self.computation_context = {
            "result": result,
            "input": (X, Y),
        }
        return self.computation_context

    def _computation_kernel(self, X, Y, x, y, z):
        conjugate = te.compute(
            (x, y, z), lambda i, j, k: -1 * Y[i, j, k], name=get_name("conjugate_input")
        )

        child_context = self._fft._set_computation_context(
            {"X": X, "Y": conjugate, "x": x, "y": y, "z": z}
        )
        fft_real, fft_imag = child_context["result"]

        result_real = te.compute(
            (x, y, z),
            lambda i, j, k: te.div(fft_real[i, j, k], z),
            name=get_name("ifft_real"),
        )

        result_imag = te.compute(
            (x, y, z),
            lambda i, j, k: te.div(-1 * fft_imag[i, j, k], z),
            name=get_name("ifft_imag"),
        )

        return result_real, result_imag
