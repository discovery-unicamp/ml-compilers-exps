from tvm import te

from tvm_te_operators.utils import get_name


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
                X[i, j, k] > X[i, j, k + 1],
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
                X[i, j, k] < X[i, j, k + 1],
            ),
            name=get_name("troughs"),
        )

        return (peaks, troughs)
