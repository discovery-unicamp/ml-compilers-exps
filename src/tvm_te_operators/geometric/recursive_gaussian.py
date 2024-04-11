import tvm.testing
from tvm import te
import numpy as np

from tvm_te_operators.utils import (
    get_name
)

class RecursiveGaussianFilter:
    def __init__(self, sigma, order=(1, 1, 1), computation_context={}):
        assert len(order) == 3
        self._filter0 = (
            DericheFilter3D(sigma=sigma, dim=0, order=order[0])
            if order[0] is not None
            else None
        )
        self._filter1 = (
            DericheFilter3D(sigma=sigma, dim=1, order=order[1])
            if order[1] is not None
            else None
        )
        self._filter2 = (
            DericheFilter3D(sigma=sigma, dim=2, order=order[2])
            if order[2] is not None
            else None
        )
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        x = computation_context.get("x", te.var("x"))
        y = computation_context.get("y", te.var("y"))
        z = computation_context.get("z", te.var("z"))
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))
        child_context = (
            self._filter0._set_computation_context({"x": x, "y": y, "z": z, "X": X})
            if self._filter0
            else {"result": X}
        )
        child_context = (
            self._filter1._set_computation_context(
                {"x": x, "y": y, "z": z, "X": child_context["result"]}
            )
            if self._filter1
            else {"result": child_context["result"]}
        )
        child_context = (
            self._filter2._set_computation_context(
                {"x": x, "y": y, "z": z, "X": child_context["result"]}
            )
            if self._filter2
            else {"result": child_context["result"]}
        )

        self.computation_context = {
            "result": child_context["result"],
            "input": X,
        }
        return self.computation_context

    @property
    def computation(self):
        return self.computation_context

    @property
    def schedule(self):
        s = te.create_schedule(self.computation_context[-1].op)
        return s

    def build(self):
        return tvm.build(
            self.schedule, self.computation, self.tgt, name="recursive_gaussian_filter"
        )


class DericheFilter:
    def __init__(self, sigma, dim, order):
        self.dim = dim
        self.order = order
        self.sigma = np.float32(sigma)
        self._n0 = np.zeros(3, dtype="float32")
        self._n1 = np.zeros(3, dtype="float32")
        self._n2 = np.zeros(3, dtype="float32")
        self._n3 = np.zeros(3, dtype="float32")
        self._d1 = np.zeros(3, dtype="float32")
        self._d2 = np.zeros(3, dtype="float32")
        self._d3 = np.zeros(3, dtype="float32")
        self._d4 = np.zeros(3, dtype="float32")
        self._makeND()

    def _makeND(self):
        a00 = 1.6797292232361107
        a10 = 3.7348298269103580
        b00 = 1.7831906544515104
        b10 = 1.7228297663338028
        c00 = -0.6802783501806897
        c10 = -0.2598300478959625
        w00 = 0.6318113174569493
        w10 = 1.9969276832487770
        a01 = 0.6494024008440620
        a11 = 0.9557370760729773
        b01 = 1.5159726670750566
        b11 = 1.5267608734791140
        c01 = -0.6472105276644291
        c11 = -4.5306923044570760
        w01 = 2.0718953658782650
        w11 = 0.6719055957689513
        a02 = 0.3224570510072559
        a12 = -1.7382843963561239
        b02 = 1.3138054926516880
        b12 = 1.2402181393295362
        c02 = -1.3312275593739595
        c12 = 3.6607035671974897
        w02 = 2.1656041357418863
        w12 = 0.7479888745408682
        a0 = [a00, a01, a02]
        a1 = [a10, a11, a12]
        b0 = [b00, b01, b02]
        b1 = [b10, b11, b12]
        c0 = [c00, c01, c02]
        c1 = [c10, c11, c12]
        w0 = [w00, w01, w02]
        w1 = [w10, w11, w12]
        sigma = self.sigma

        for i in range(3):
            n0 = (a0[i] + c0[i]) if i % 2 == 0 else 0.0
            n1 = np.exp(-b1[i] / sigma) * (
                c1[i] * np.sin(w1[i] / sigma)
                - (c0[i] + 2.0 * a0[i]) * np.cos(w1[i] / sigma)
            ) + np.exp(-b0[i] / sigma) * (
                a1[i] * np.sin(w0[i] / sigma)
                - (2.0 * c0[i] + a0[i]) * np.cos(w0[i] / sigma)
            )
            n2 = (
                2.0
                * np.exp(-(b0[i] + b1[i]) / sigma)
                * (
                    (a0[i] + c0[i]) * np.cos(w1[i] / sigma) * np.cos(w0[i] / sigma)
                    - a1[i] * np.cos(w1[i] / sigma) * np.sin(w0[i] / sigma)
                    - c1[i] * np.cos(w0[i] / sigma) * np.sin(w1[i] / sigma)
                )
                + c0[i] * np.exp(-2.0 * b0[i] / sigma)
                + a0[i] * np.exp(-2.0 * b1[i] / sigma)
            )
            n3 = np.exp(-(b1[i] + 2.0 * b0[i]) / sigma) * (
                c1[i] * np.sin(w1[i] / sigma) - c0[i] * np.cos(w1[i] / sigma)
            ) + np.exp(-(b0[i] + 2.0 * b1[i]) / sigma) * (
                a1[i] * np.sin(w0[i] / sigma) - a0[i] * np.cos(w0[i] / sigma)
            )
            d1 = -2.0 * np.exp(-b0[i] / sigma) * np.cos(w0[i] / sigma) - 2.0 * np.exp(
                -b1[i] / sigma
            ) * np.cos(w1[i] / sigma)
            d2 = (
                4.0
                * np.exp(-(b0[i] + b1[i]) / sigma)
                * np.cos(w0[i] / sigma)
                * np.cos(w1[i] / sigma)
                + np.exp(-2.0 * b0[i] / sigma)
                + np.exp(-2.0 * b1[i] / sigma)
            )
            d3 = -2.0 * np.exp(-(b0[i] + 2.0 * b1[i]) / sigma) * np.cos(
                w0[i] / sigma
            ) - 2.0 * np.exp(-(b1[i] + 2.0 * b0[i]) / sigma) * np.cos(w1[i] / sigma)
            d4 = np.exp(-2.0 * (b0[i] + b1[i]) / sigma)
            self._n0[i] = np.float32(n0)
            self._n1[i] = np.float32(n1)
            self._n2[i] = np.float32(n2)
            self._n3[i] = np.float32(n3)
            self._d1[i] = np.float32(d1)
            self._d2[i] = np.float32(d2)
            self._d3[i] = np.float32(d3)
            self._d4[i] = np.float32(d4)
        self._scaleN()

    def _scaleN(self):
        sigma = self.sigma
        n = 1 + 2 * int(10.0 * sigma)
        x = np.zeros(n, dtype=np.float32)
        y0 = np.zeros(n, dtype=np.float32)
        y1 = np.zeros(n, dtype=np.float32)
        y2 = np.zeros(n, dtype=np.float32)
        m = int((n - 1) / 2)
        x[m] = 1.0
        y0 = self.applyN(x, 0)
        y1 = self.applyN(x, 1)
        y2 = self.applyN(x, 2)
        s = np.zeros(3, np.float64)
        j = n - 1
        for i in range(n):
            if i >= j:
                break
            t = i - m
            s[0] += y0[j] + y0[i]
            s[1] += np.sin(t / sigma) * (y1[j] - y1[i])
            s[2] += np.cos(t * np.sqrt(2.0) / sigma) * (y2[j] + y2[i])
            j -= 1
        s[0] += y0[m]
        s[2] += y2[m]
        s[1] *= sigma * np.exp(0.5)
        s[2] *= -(sigma * sigma) / 2.0 * np.exp(1.0)
        for i in range(3):
            self._n0[i] /= s[i]
            self._n1[i] /= s[i]
            self._n2[i] /= s[i]
            self._n3[i] /= s[i]

    def applyN(self, x, nd, y=None, xp=np):
        if y is None:
            y = xp.zeros(x.shape, dtype=x.dtype)
        applyN(
            x,
            y,
            nd,
            *x.shape,
            self._n0[nd],
            self._n1[nd],
            self._n2[nd],
            self._n3[nd],
            self._d1[nd],
            self._d2[nd],
            self._d3[nd],
            self._d4[nd],
        )
        return y


def applyN(x, y, nd, m, n0, n1, n2, n3, d1, d2, d3, d4):
    yim4 = np.float32(0.0)
    yim3 = np.float32(0.0)
    yim2 = np.float32(0.0)
    yim1 = np.float32(0.0)
    xim3 = np.float32(0.0)
    xim2 = np.float32(0.0)
    xim1 = np.float32(0.0)
    for i in range(m):
        xi = x[i]
        yi = (
            n0 * xi
            + n1 * xim1
            + n2 * xim2
            + n3 * xim3
            - d1 * yim1
            - d2 * yim2
            - d3 * yim3
            - d4 * yim4
        )
        y[i] = yi
        yim4 = yim3
        yim3 = yim2
        yim2 = yim1
        yim1 = yi
        xim3 = xim2
        xim2 = xim1
        xim1 = xi
    n1 = n1 - d1 * n0
    n2 = n2 - d2 * n0
    n3 = n3 - d3 * n0
    n4 = -d4 * n0
    if nd % 2 != 0:
        n1 = -n1
        n2 = -n2
        n3 = -n3
        n4 = -n4
    yip4 = np.float32(0.0)
    yip3 = np.float32(0.0)
    yip2 = np.float32(0.0)
    yip1 = np.float32(0.0)
    xip4 = np.float32(0.0)
    xip3 = np.float32(0.0)
    xip2 = np.float32(0.0)
    xip1 = np.float32(0.0)

    for i in range(m - 1, -1, -1):
        xi = x[i]
        yi = (
            n1 * xip1
            + n2 * xip2
            + n3 * xip3
            + n4 * xip4
            - d1 * yip1
            - d2 * yip2
            - d3 * yip3
            - d4 * yip4
        )
        y[i] += yi
        yip4 = yip3
        yip3 = yip2
        yip2 = yip1
        yip1 = yi
        xip4 = xip3
        xip3 = xip2
        xip2 = xip1
        xip1 = xi


class DericheFilter3D(DericheFilter):
    def __init__(self, sigma, dim, order, computation_context={}):
        super().__init__(sigma, dim, order)
        self._set_computation_context(computation_context)

    def _set_computation_context(self, computation_context):
        nd = self.order
        x = computation_context.get("x", 64)
        y = computation_context.get("y", 64)
        z = computation_context.get("z", 256)
        X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

        method = self.__getattribute__(f"_computation_{self.dim}")

        result = method(
            X,
            x,
            y,
            z,
            self.order,
            self._n0[nd],
            self._n1[nd],
            self._n2[nd],
            self._n3[nd],
            self._d1[nd],
            self._d2[nd],
            self._d3[nd],
            self._d4[nd],
        )
        self.computation_context = {
            "result": result,
            "input": X,
        }
        return self.computation_context

    @property
    def computation(self):
        return self.computation_context

    def _computation_kernel(self, X, x, y, z, order, n0, n1, n2, n3, d1, d2, d3, d4):
        # Forward sweep
        f_sweep_state = te.placeholder((x, y, z), dtype=X.dtype, name=get_name("f_sweep_state"))
        f_sweep_init = te.compute(
            (1, y, z), lambda _, j, k: n0 * X[0, j, k], name=get_name("foward_sweep_init")
        )

        f_sweep_update = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                i >= 4,
                n0 * X[i, j, k]
                + n1 * X[i - 1, j, k]
                + n2 * X[i - 2, j, k]
                + n3 * X[i - 3, j, k]
                - d1 * f_sweep_state[i - 1, j, k]
                - d2 * f_sweep_state[i - 2, j, k]
                - d3 * f_sweep_state[i - 3, j, k]
                - d4 * f_sweep_state[i - 4, j, k],
                te.if_then_else(
                    i == 3,
                    n0 * X[i, j, k]
                    + n1 * X[i - 1, j, k]
                    + n2 * X[i - 2, j, k]
                    + n3 * X[i - 3, j, k]
                    - d1 * f_sweep_state[i - 1, j, k]
                    - d2 * f_sweep_state[i - 2, j, k]
                    - d3 * f_sweep_state[i - 3, j, k],
                    te.if_then_else(
                        i == 2,
                        n0 * X[i, j, k]
                        + n1 * X[i - 1, j, k]
                        + n2 * X[i - 2, j, k]
                        - d1 * f_sweep_state[i - 1, j, k]
                        - d2 * f_sweep_state[i - 2, j, k],
                        n0 * X[i, j, k]
                        + n1 * X[i - 1, j, k]
                        - d1 * f_sweep_state[i - 1, j, k]
                    )
                )
            ),
            name=get_name("foward_sweep_update"),
        )

        f_sweep = tvm.te.scan(
            f_sweep_init,
            f_sweep_update,
            f_sweep_state,
            inputs=[X],
            name=get_name("forward_scan"),
        )

        # Reverse sweep
        n1 = n1 - d1 * n0
        n2 = n2 - d2 * n0
        n3 = n3 - d3 * n0
        n4 = -d4 * n0
        if order % 2 != 0:
            n1 = -n1
            n2 = -n2
            n3 = -n3
            n4 = -n4
        r_sweep_state = te.placeholder((x, y, z), dtype=X.dtype, name=get_name("r_sweep_state"))
        r_sweep_init = te.compute(
            (1, y, z),
            lambda _, j, k: np.float64(0.0).astype(X.dtype),
            name=get_name("reverse_sweep_init"),
        )
        r_sweep_update = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                i >= 4,
                n1 * X[x - (i + 1) + 1, j, k]
                + n2 * X[x - (i + 1) + 2 , j, k]
                + n3 * X[x - (i + 1) + 3, j, k]
                + n4 * X[x - (i + 1) + 4, j, k]
                - d1 * r_sweep_state[i - 1, j, k]
                - d2 * r_sweep_state[i - 2, j, k]
                - d3 * r_sweep_state[i - 3, j, k]
                - d4 * r_sweep_state[i - 4, j, k],
                te.if_then_else(
                    i == 3,
                    n1 * X[x - (i + 1) + 1, j, k]
                    + n2 * X[x - (i + 1) + 2 , j, k]
                    + n3 * X[x - (i + 1) + 3, j, k]
                    - d1 * r_sweep_state[i - 1, j, k]
                    - d2 * r_sweep_state[i - 2, j, k]
                    - d3 * r_sweep_state[i - 3, j, k],
                    te.if_then_else(
                    i == 2,
                    n1 * X[x - (i + 1) + 1, j, k]
                    + n2 * X[x - (i + 1) + 2 , j, k]
                    - d1 * r_sweep_state[i - 1, j, k]
                    - d2 * r_sweep_state[i - 2, j, k],
                    n1 * X[x - (i + 1) + 1, j, k]
                    - d1 * r_sweep_state[i - 1, j, k],
                    )
                ),
            ),
            name=get_name("reverse_sweep_update"),
        )

        r_sweep = tvm.te.scan(
            r_sweep_init,
            r_sweep_update,
            r_sweep_state,
            inputs=[X],
            name=get_name("reverse_scan"),
        )

        # Join Forward and Reverse Sweep Values
        result = te.compute(
            (x, y, z),
            lambda i, j, k: f_sweep[i, j, k] + r_sweep[x - (i + 1), j, k],
            get_name("join_sweeps"),
        )

        return result

    def _computation_0(self, X, x, y, z, order, n0, n1, n2, n3, d1, d2, d3, d4):
        X = te.compute((z, y, x), lambda i, j, k: X[k, j, i], name=get_name("reshape"))
        X = self._computation_kernel(X, z, y, x, order, n0, n1, n2, n3, d1, d2, d3, d4)
        result = te.compute(
            (x, y, z), lambda i, j, k: X[k, j, i], name=get_name("reshape_back")
        )
        return result

    def _computation_1(self, X, x, y, z, order, n0, n1, n2, n3, d1, d2, d3, d4):
        X = te.compute((y, x, z), lambda i, j, k: X[j, i, k], name=get_name("reshape"))
        X = self._computation_kernel(X, y, x, z, order, n0, n1, n2, n3, d1, d2, d3, d4)
        result = te.compute(
            (x, y, z), lambda i, j, k: X[j, i, k], name=get_name("reshape_back")
        )
        return result

    def _computation_2(self, X, x, y, z, order, n0, n1, n2, n3, d1, d2, d3, d4):
        return self._computation_kernel(
            X, x, y, z, order, n0, n1, n2, n3, d1, d2, d3, d4
        )

    
