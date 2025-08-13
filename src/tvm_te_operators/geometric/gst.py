import tvm.testing
from tvm import te
import numpy as np

from tvm_te_operators.geometric.recursive_gaussian import RecursiveGaussianFilter
from tvm_te_operators.utils import get_name


class GradientStructureTensor:
    def __init__(
        self, sigma=(3, 3, 3), gradient_smoothing=(1, 1, 1), computation_context={}
    ):
        self._rgfSmoother1 = (
            RecursiveGaussianFilter(sigma[0], order=(0, None, None))
            if sigma[0] >= 1
            else None
        )
        self._rgfSmoother2 = (
            RecursiveGaussianFilter(sigma[1], order=(None, 0, None))
            if sigma[1] >= 1
            else None
        )
        self._rgfSmoother3 = (
            RecursiveGaussianFilter(sigma[2], order=(None, None, 0))
            if sigma[2] >= 1
            else None
        )

        self._rgfGradient1 = RecursiveGaussianFilter(gradient_smoothing[0], (1, 0, 0))
        self._rgfGradient2 = RecursiveGaussianFilter(gradient_smoothing[1], (0, 1, 0))
        self._rgfGradient3 = RecursiveGaussianFilter(gradient_smoothing[2], (0, 0, 1))

        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", 64)
        y = computation_context.get("y", 64)
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        gi = self._rgfGradient1._set_computation_context(
            {"x": x, "y": y, "z": z, "X": X}
        )["result"]
        gj = self._rgfGradient2._set_computation_context(
            {"x": x, "y": y, "z": z, "X": X}
        )["result"]
        gk = self._rgfGradient3._set_computation_context(
            {"x": x, "y": y, "z": z, "X": X}
        )["result"]

        gradients = [
            te.compute(
                (x, y, z),
                lambda i, j, k: gi[i, j, k] * gi[i, j, k],
                name=get_name("gi2"),
            ),
            te.compute(
                (x, y, z),
                lambda i, j, k: gj[i, j, k] * gj[i, j, k],
                name=get_name("gj2"),
            ),
            te.compute(
                (x, y, z),
                lambda i, j, k: gk[i, j, k] * gk[i, j, k],
                name=get_name("gk2"),
            ),
            te.compute(
                (x, y, z),
                lambda i, j, k: gi[i, j, k] * gj[i, j, k],
                name=get_name("gigj"),
            ),
            te.compute(
                (x, y, z),
                lambda i, j, k: gi[i, j, k] * gk[i, j, k],
                name=get_name("gigk"),
            ),
            te.compute(
                (x, y, z),
                lambda i, j, k: gj[i, j, k] * gk[i, j, k],
                name=get_name("gjgk"),
            ),
        ]

        for i in range(len(gradients)):
            g = gradients[i]
            if self._rgfSmoother1:
                g = self._rgfSmoother1._set_computation_context(
                    {"x": x, "y": y, "z": z, "X": g}
                )["result"]
            if self._rgfSmoother2:
                g = self._rgfSmoother2._set_computation_context(
                    {"x": x, "y": y, "z": z, "X": g}
                )["result"]
            if self._rgfSmoother3:
                g = self._rgfSmoother3._set_computation_context(
                    {"x": x, "y": y, "z": z, "X": g}
                )["result"]
            gradients[i] = g

        self.computation_context = {
            "result": gradients,
            "input": X,
        }
        return self.computation_context


    @property
    def computation(self):
        return self.computation_context



