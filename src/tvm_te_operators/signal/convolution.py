from tvm import te

from tvm_te_operators.utils import (
    get_name,
)

class Convolution1D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 256)
        X = computation_context.get("X", te.placeholder((x,), name=get_name("X")))
        w = computation_context.get("w", 3)
        W = computation_context.get("W", te.placeholder((w,), name=get_name("W")))

        result = self._computation_kernel(
            X,
            x,
            W,
            w,
        )

        self.computation_context = {
            "result": result,
            "input": (X,W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, W, w):
        pad = w // 2
        # Pad input
        Xpad = te.compute(
            (x + 2 * pad,),
            lambda i: te.if_then_else(
                te.all(i >= pad, i - pad < x),
                X[i - pad],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rx = te.reduce_axis((0, w), name="rx")
        # Compute the convolution
        res = te.compute(
            (x,),
            lambda i: te.sum(
                Xpad[i + rx] * W[w-rx], axis=[rx]
            ),
            name=get_name("conv1d"),
        )
        return res
    

class Convolution2D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x1 = computation_context.get("x1", 64)
        x2 = computation_context.get("x2", 256)
        X = computation_context.get("X", te.placeholder((x1,x2), name=get_name("X")))
        w1 = computation_context.get("w1", 3)
        w2 = computation_context.get("w2", 3)
        W = computation_context.get("W", te.placeholder((w1,w2), name=get_name("W")))

        result = self._computation_kernel(
            X,
            x1,
            x2,
            W,
            w1,
            w2,
        )

        self.computation_context = {
            "result": result,
            "input": (X,W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x1, x2, W, w1, w2):
        pad = (w1 // 2, w2 // 2)
        # Pad input
        Xpad = te.compute(
            (x1 + 2 * pad[0], x2 + 2 * pad[1]),
            lambda i,j: te.if_then_else(
                te.all(i >= pad[0], i - pad[0] < x1, j >= pad[1], j - pad[1] < x2),
                X[i - pad[0], j - pad[1]],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rx1 = te.reduce_axis((0, w1), name="rx1")
        rx2 = te.reduce_axis((0, w2), name="rx2")
        # Compute the convolution
        res = te.compute(
            (x1,x2),
            lambda i, j: te.sum(
                Xpad[i + rx1, j + rx2] * W[w1-rx1, w2-rx2], axis=[rx1, rx2]
            ),
            name=get_name("conv2d"),
        )
        return res
    
class Correlation1D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 256)
        X = computation_context.get("X", te.placeholder((x,), name=get_name("X")))
        w = computation_context.get("w", 3)
        W = computation_context.get("W", te.placeholder((w,), name=get_name("W")))

        result = self._computation_kernel(
            X,
            x,
            W,
            w,
        )

        self.computation_context = {
            "result": result,
            "input": (X,W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x, W, w):
        pad = w // 2
        # Pad input
        Xpad = te.compute(
            (x + 2 * pad,),
            lambda i: te.if_then_else(
                te.all(i >= pad, i - pad < x),
                X[i - pad],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rx = te.reduce_axis((0, w), name="rx")
        # Compute the convolution
        res = te.compute(
            (x,),
            lambda i: te.sum(
                Xpad[i + rx] * W[rx], axis=[rx]
            ),
            name=get_name("cor1d"),
        )
        return res
    

class Correlation2D:
    def __init__(self, computation_context={}):
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x1 = computation_context.get("x1", 64)
        x2 = computation_context.get("x2", 256)
        X = computation_context.get("X", te.placeholder((x1,x2), name=get_name("X")))
        w1 = computation_context.get("w1", 3)
        w2 = computation_context.get("w2", 3)
        W = computation_context.get("W", te.placeholder((w1,w2), name=get_name("W")))

        result = self._computation_kernel(
            X,
            x1,
            x2,
            W,
            w1,
            w2,
        )

        self.computation_context = {
            "result": result,
            "input": (X,W),
        }
        return self.computation_context

    def _computation_kernel(self, X, x1, x2, W, w1, w2):
        pad = (w1 // 2, w2 // 2)
        # Pad input
        Xpad = te.compute(
            (x1 + 2 * pad[0], x2 + 2 * pad[1]),
            lambda i,j: te.if_then_else(
                te.all(i >= pad[0], i - pad[0] < x1, j >= pad[1], j - pad[1] < x2),
                X[i - pad[0], j - pad[1]],
                te.const(0.0, X.dtype),
            ),
            name=get_name("Xpad"),
        )
        # Create reduction variables
        rx1 = te.reduce_axis((0, w1), name="rx1")
        rx2 = te.reduce_axis((0, w2), name="rx2")
        # Compute the convolution
        res = te.compute(
            (x1,x2),
            lambda i, j: te.sum(
                Xpad[i + rx1, j + rx2] * W[rx1, rx2], axis=[rx1, rx2]
            ),
            name=get_name("cor2d"),
        )
        return res