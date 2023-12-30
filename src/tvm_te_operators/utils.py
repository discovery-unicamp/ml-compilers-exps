from uuid import uuid4

from tvm import te
import tvm.topi as topi
import numpy as np


def get_name(name):
    return f"{name}-{uuid4()}"


class FirstDerivative:
    def __init__(self, axis=2, computation_context={}):
        self._axis = axis  # values: 0, 1 or 2. TODO: add assert
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        result = getattr(self, f"_computation_kernel_{self._axis}")(
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

    def _computation_kernel_2(self, X, x, y, z):
        apply_1d = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                k > 0,
                te.if_then_else(
                    k < z - 1,
                    X[i, j, k - 1] * -0.5 + X[i, j, k + 1] * 0.5,
                    X[i, j, k - 1] * -0.5 + X[i, j, k] * 0.5,
                ),
                te.if_then_else(
                    k < z - 1,
                    X[i, j, k] * -0.5 + X[i, j, k + 1] * 0.5,
                    X[i, j, k] * -0.5 + X[i, j, k] * 0.5,
                ),
            ),
            name=get_name("apply_1d"),
        )

        smooth_1 = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                i > 0,
                te.if_then_else(
                    i < x - 1,
                    apply_1d[i - 1, j, k] * 0.178947
                    + apply_1d[i, j, k] * 0.642105
                    + apply_1d[i + 1, j, k] * 0.178947,
                    apply_1d[i - 1, j, k] * 0.178947
                    + apply_1d[i, j, k] * 0.642105
                    + apply_1d[i, j, k] * 0.178947,
                ),
                te.if_then_else(
                    i < x - 1,
                    apply_1d[i, j, k] * 0.178947
                    + apply_1d[i, j, k] * 0.642105
                    + apply_1d[i + 1, j, k] * 0.178947,
                    apply_1d[i, j, k] * 0.178947
                    + apply_1d[i, j, k] * 0.642105
                    + apply_1d[i, j, k] * 0.178947,
                ),
            ),
            name=get_name("smooth_1"),
        )

        smooth_2 = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                j > 0,
                te.if_then_else(
                    j < y - 1,
                    smooth_1[i, j - 1, k] * 0.178947
                    + smooth_1[i, j, k] * 0.642105
                    + smooth_1[i, j + 1, k] * 0.178947,
                    smooth_1[i, j - 1, k] * 0.178947
                    + smooth_1[i, j, k] * 0.642105
                    + smooth_1[i, j, k] * 0.178947,
                ),
                te.if_then_else(
                    j < y - 1,
                    smooth_1[i, j, k] * 0.178947
                    + smooth_1[i, j, k] * 0.642105
                    + smooth_1[i, j + 1, k] * 0.178947,
                    smooth_1[i, j, k] * 0.178947
                    + smooth_1[i, j, k] * 0.642105
                    + smooth_1[i, j, k] * 0.178947,
                ),
            ),
            name=get_name("smooth_2"),
        )

        return (smooth_2,)


class LocalEvents:
    def __init__(self, computation_context={}):
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
        peaks = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                k > 0,
                te.if_then_else(
                    k < z - 1,
                    X[i, j, k] > X[i, j, k + 1] and X[i, j, k] > X[i, j, k - 1],
                    X[i, j, k] > X[i, j, k - 1],
                ),
                te.if_then_else(
                    k < z - 1,
                    X[i, j, k] > X[i, j, k + 1],
                    1,
                ),
            ),
            name=get_name("peaks"),
        )

        troughs = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                k > 0,
                te.if_then_else(
                    k < z - 1,
                    X[i, j, k] < X[i, j, k + 1] and X[i, j, k] < X[i, j, k - 1],
                    X[i, j, k] < X[i, j, k - 1],
                ),
                te.if_then_else(
                    k < z - 1,
                    X[i, j, k] < X[i, j, k + 1],
                    1,
                ),
            ),
            name=get_name("troughs"),
        )

        return (peaks, troughs)


class CumSum:
    def __init__(self, axis=2, computation_context={}):
        self._axis = axis  # values: 0, 1 or 2. TODO: add assert
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        result = getattr(self, f"_computation_kernel_{self._axis}")(
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

    def _computation_kernel_0(self, X, x, y, z):
        cumsum = te.compute(
            (1, y, z), lambda i, j, k: X[i, j, k], name=get_name("step_0")
        )
        for step in range(1, x):
            aux = te.compute(
                (step + 1, y, z),
                lambda i, j, k: te.if_then_else(
                    i < step, cumsum[i, j, k], cumsum[i - 1, j, k] + X[i, j, k]
                ),
                name=get_name(f"step_{step}"),
            )
            cumsum = aux

        return (cumsum,)

    def _computation_kernel_1(self, X, x, y, z):
        cumsum = te.compute(
            (x, 1, z), lambda i, j, k: X[i, j, k], name=get_name("step_0")
        )
        for step in range(1, y):
            aux = te.compute(
                (x, step + 1, z),
                lambda i, j, k: te.if_then_else(
                    j < step, cumsum[i, j, k], cumsum[i, j - 1, k] + X[i, j, k]
                ),
                name=get_name(f"step_{step}"),
            )
            cumsum = aux

        return (cumsum,)

    def _computation_kernel_2(self, X, x, y, z):
        cumsum = X
        for step in range(0, z):
            aux = te.compute(
                (x, y, z),
                lambda i, j, k: te.if_then_else(
                    k <= step, cumsum[i, j, k], cumsum[i, j, k] + X[i, j, step]
                ),
                name=get_name(f"step_{step}"),
            )
            cumsum = aux

        return (cumsum,)


class Unwrap:
    def __init__(self, discont=None, axis=2, period=2 * np.pi, computation_context={}):
        self._axis = axis  # values: 0, 1 or 2. TODO: add assert
        self._dicont = discont if discont is not None else period / 2
        self._period = period
        self._cumsum = CumSum(axis=2)
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 1)
        y = computation_context.get("y", 1)
        z = computation_context.get("z", 64)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        result = getattr(self, f"_computation_kernel_{self._axis}")(
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

    def _computation_kernel_2(self, X, x, y, z):
        period = te.const(self._period, X.dtype)
        interval_high = te.const(self._period / 2, X.dtype)
        interval_low = te.const(-self._period / 2, X.dtype)
        discont = te.const(self._dicont, X.dtype)
        dd = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(k == 0, 0, X[i, j, k + 1] - X[i, j, k]),
            name=get_name("dd"),
        )

        ddmod = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                dd[i, j, k] - interval_low >= 0,
                te.fmod(dd[i, j, k] - interval_low, period) + interval_low,
                te.fmod(dd[i, j, k] - interval_low, period) + interval_low + period,
            ),
            name=get_name("ddmod"),
        )

        boundary_amb = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                te.all(ddmod[i, j, k] == interval_low, dd[i, j, k] > 0),
                interval_high,
                ddmod[i, j, k],
            ),
            name=get_name("boundary_amb"),
        )

        ph_correct = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                te.abs(dd[i, j, k]) < discont,
                0,
                boundary_amb[i, j, k] - dd[i, j, k],
            ),
            name=get_name("boundary_amb"),
        )

        ph_correct_cumsum = self._cumsum._set_computation_context(
            {"X": ph_correct, "x": x, "y": y, "z": z}
        )["result"][0]

        result = te.compute(
            (x, y, z),
            lambda i, j, k: X[i, j, k] + ph_correct_cumsum[i, j, k],
            name=get_name("unwrap"),
        )

        return (result,)
