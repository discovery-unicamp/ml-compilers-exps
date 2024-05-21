from tvm import te

from tvm_te_operators.utils import (
    get_name,
)


class Convolution1D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = 1
        y = 1
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        w = computation_context.get("w", 3)
        W = computation_context.get(
            "W", te.placeholder((1, 1, w), dtype=X.dtype, name=get_name("W"))
        )

        result = self._computation_kernel(
            X,
            x,
            y,
            z,
            W,
            w,
        )

        self.computation_context = {
            "result": result,
            "input": (X, W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z, W, w):
        pad = w // 2
        # Pad input
        Xpad = te.compute(
            (
                x,
                y,
                z + 2 * pad,
            ),
            lambda i, j, k: te.if_then_else(
                te.all(k >= pad, k - pad < z),
                X[i, j, k - pad],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rz = te.reduce_axis((0, w), name="rz")
        # Compute the convolution
        res = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(Xpad[i, j, k + rz] * W[0, 0, w - rz - 1], axis=[rz]),
            name=get_name("conv1d"),
        )
        return (res,)


class Convolution2D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = 1
        y = computation_context.get("y", 64)
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        w1 = computation_context.get("w1", 3)
        w2 = computation_context.get("w2", 3)
        W = computation_context.get(
            "W", te.placeholder((1, w1, w2), dtype=X.dtype, name=get_name("W"))
        )

        result = self._computation_kernel(
            X,
            x,
            y,
            z,
            W,
            w1,
            w2,
        )

        self.computation_context = {
            "result": result,
            "input": (X, W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z, W, w1, w2):
        pad = (w1 // 2, w2 // 2)
        # Pad input
        Xpad = te.compute(
            (x, y + 2 * pad[0], z + 2 * pad[1]),
            lambda i, j, k: te.if_then_else(
                te.all(j >= pad[0], j - pad[0] < y, k >= pad[1], k - pad[1] < z),
                X[i, j - pad[0], k - pad[1]],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        ry = te.reduce_axis((0, w1), name="ry")
        rz = te.reduce_axis((0, w2), name="rz")
        # Compute the convolution
        res = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                Xpad[i, j + ry, k + rz] * W[0, w1 - ry - 1, w2 - rz - 1], axis=[ry, rz]
            ),
            name=get_name("conv2d"),
        )
        return (res,)


class Convolution3D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 64)
        y = computation_context.get("y", 64)
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        w1 = computation_context.get("w1", 3)
        w2 = computation_context.get("w2", 3)
        w3 = computation_context.get("w3", 3)
        W = computation_context.get(
            "W", te.placeholder((w1, w2, w3), dtype=X.dtype, name=get_name("W"))
        )

        result = self._computation_kernel(X, x, y, z, W, w1, w2, w3)

        self.computation_context = {
            "result": result,
            "input": (X, W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z, W, w1, w2, w3):
        pad = (w1 // 2, w2 // 2, w3 // 2)
        # Pad input
        Xpad = te.compute(
            (x + 2 * pad[0], y + 2 * pad[1], z + 2 * pad[2]),
            lambda i, j, k: te.if_then_else(
                te.all(
                    i >= pad[0],
                    i - pad[0] < x,
                    j >= pad[1],
                    j - pad[1] < y,
                    k >= pad[2],
                    k - pad[2] < z,
                ),
                X[i - pad[0], j - pad[1], k - pad[2]],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rx = te.reduce_axis((0, w1), name="rx")
        ry = te.reduce_axis((0, w2), name="ry")
        rz = te.reduce_axis((0, w3), name="rz")
        # Compute the convolution
        res = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                Xpad[i + rx, j + ry, k + rz] * W[w1 - rx - 1, w2 - ry - 1, w3 - rz - 1],
                axis=[rx, ry, rz],
            ),
            name=get_name("conv3d"),
        )
        return (res,)


class Correlation1D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = 1
        y = 1
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        w = computation_context.get("w", 3)
        W = computation_context.get(
            "W", te.placeholder((1, 1, w), dtype=X.dtype, name=get_name("W"))
        )

        result = self._computation_kernel(
            X,
            x,
            y,
            z,
            W,
            w,
        )

        self.computation_context = {
            "result": result,
            "input": (X, W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z, W, w):
        pad = w // 2
        # Pad input
        Xpad = te.compute(
            (
                x,
                y,
                z + 2 * pad,
            ),
            lambda i, j, k: te.if_then_else(
                te.all(k >= pad, k - pad < z),
                X[i, j, k - pad],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rz = te.reduce_axis((0, w), name="rz")
        # Compute the Correlation
        res = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(Xpad[i, j, k + rz] * W[0, 0, rz], axis=[rz]),
            name=get_name("cor1d"),
        )
        return (res,)


class Correlation2D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = 1
        y = computation_context.get("y", 64)
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        w1 = computation_context.get("w1", 3)
        w2 = computation_context.get("w2", 3)
        W = computation_context.get(
            "W", te.placeholder((1, w1, w2), dtype=X.dtype, name=get_name("W"))
        )

        result = self._computation_kernel(
            X,
            x,
            y,
            z,
            W,
            w1,
            w2,
        )

        self.computation_context = {
            "result": result,
            "input": (X, W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z, W, w1, w2):
        pad = (w1 // 2, w2 // 2)
        # Pad input
        Xpad = te.compute(
            (x, y + 2 * pad[0], z + 2 * pad[1]),
            lambda i, j, k: te.if_then_else(
                te.all(j >= pad[0], j - pad[0] < y, k >= pad[1], k - pad[1] < z),
                X[i, j - pad[0], k - pad[1]],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        ry = te.reduce_axis((0, w1), name="ry")
        rz = te.reduce_axis((0, w2), name="rz")
        # Compute the Correlation
        res = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                Xpad[i, j + ry, k + rz] * W[0, ry, rz], axis=[ry, rz]
            ),
            name=get_name("cor2d"),
        )
        return (res,)


class Correlation3D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 64)
        y = computation_context.get("y", 64)
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        w1 = computation_context.get("w1", 3)
        w2 = computation_context.get("w2", 3)
        w3 = computation_context.get("w3", 3)
        W = computation_context.get(
            "W", te.placeholder((w1, w2, w3), dtype=X.dtype, name=get_name("W"))
        )

        result = self._computation_kernel(X, x, y, z, W, w1, w2, w3)

        self.computation_context = {
            "result": result,
            "input": (X, W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, y, z, W, w1, w2, w3):
        pad = (w1 // 2, w2 // 2, w3 // 2)
        # Pad input
        Xpad = te.compute(
            (x + 2 * pad[0], y + 2 * pad[1], z + 2 * pad[2]),
            lambda i, j, k: te.if_then_else(
                te.all(
                    i >= pad[0],
                    i - pad[0] < x,
                    j >= pad[1],
                    j - pad[1] < y,
                    k >= pad[2],
                    k - pad[2] < z,
                ),
                X[i - pad[0], j - pad[1], k - pad[2]],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rx = te.reduce_axis((0, w1), name="rx")
        ry = te.reduce_axis((0, w2), name="ry")
        rz = te.reduce_axis((0, w3), name="rz")
        # Compute the Correlation
        res = te.compute(
            (x, y, z),
            lambda i, j, k: te.sum(
                Xpad[i + rx, j + ry, k + rz] * W[rx, ry, rz], axis=[rx, ry, rz]
            ),
            name=get_name("cor3d"),
        )
        return (res,)
