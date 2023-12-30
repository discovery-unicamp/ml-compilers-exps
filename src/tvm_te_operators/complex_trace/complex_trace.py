from tvm import te, tir
import tvm.topi as topi
import numpy as np

from tvm_te_operators.complex_trace.fft import FFT, IFFT
from tvm_te_operators.utils import (
    get_name,
    FirstDerivative,
    Unwrap,
    # LocalEvents,
    # CumSum,
)


class Hilbert:
    def __init__(self, computation_context={}):
        self._fft = FFT()
        self._ifft = IFFT()
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
        fft_real, fft_imag = self._fft._set_computation_context(
            {"X": X, "x": x, "y": y, "z": z}
        )["result"]
        if z % 2 == 0:
            h = te.compute(
                (z,),
                lambda i: te.if_then_else(
                    i < te.indexdiv(z, 2), te.if_then_else(i == 0, 1, 2), 0
                ),
                name=get_name("h"),
            )
        else:
            h = te.compute(
                (z,),
                lambda i: te.if_then_else(
                    i < te.indexdiv(z + 1, 2),
                    te.if_then_else(i == 0, 1, 2),
                    0,
                ),
                name=get_name("h"),
            )

        apply_sign_real = te.compute(
            (x, y, z),
            lambda i, j, k: h[k] * fft_real[i, j, k],
            name=get_name("apply_sign_real"),
        )

        apply_sign_imag = te.compute(
            (x, y, z),
            lambda i, j, k: h[k] * fft_imag[i, j, k],
            name=get_name("apply_sign_imag"),
        )

        _, imag = self._ifft._set_computation_context(
            {"X": apply_sign_real, "Y": apply_sign_imag, "x": x, "y": y, "z": z}
        )["result"]

        real = te.compute((x, y, z), lambda i, j, k: X[i, j, k], name=get_name("real"))

        return real, imag


class Envelope:
    def __init__(self, computation_context={}):
        self._hilbert = Hilbert()
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
        hilbert_real, hilbert_imag = self._hilbert._set_computation_context(
            {"X": X, "x": x, "y": y, "z": z}
        )["result"]

        envelope = te.compute(
            (x, y, z),
            lambda i, j, k: te.sqrt(
                te.power(hilbert_real[i, j, k], 2) + te.power(hilbert_imag[i, j, k], 2)
            ),
            name=get_name("envelope"),
        )

        return (envelope,)


class InstantaneousPhaseRadians:
    def __init__(self, computation_context={}):
        self._hilbert = Hilbert()
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
        hilbert_real, hilbert_imag = self._hilbert._set_computation_context(
            {"X": X, "x": x, "y": y, "z": z}
        )["result"]

        phase = te.compute(
            (x, y, z),
            lambda i, j, k: tir.atan2(hilbert_imag[i, j, k], hilbert_real[i, j, k]),
            name=get_name("phase"),
        )

        return (phase,)


class InstantaneousPhase:
    def __init__(self, computation_context={}):
        self._rad = InstantaneousPhaseRadians()
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
        radians = self._rad._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]
        pi_const = te.const(np.pi, X.dtype)
        phase = te.compute(
            (x, y, z),
            lambda i, j, k: te.div(radians[i, j, k] * 180, pi_const),
            name=get_name("phase"),
        )

        return (phase,)


class CosineInstantaneousPhase:
    def __init__(self, computation_context={}):
        self._phase = InstantaneousPhaseRadians()
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
        phase = self._phase._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        cosine = te.compute(
            (x, y, z),
            lambda i, j, k: te.cos(phase[i, j, k]),
            name=get_name("cosine"),
        )

        return (cosine,)


class RelativeAmplitudeChange:
    def __init__(self, computation_context={}):
        self._envelope = Envelope()
        self._first_derivative = FirstDerivative()
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
        env = self._envelope._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]
        env_prime = self._first_derivative._set_computation_context(
            {"X": env, "x": x, "y": y, "z": z}
        )["result"][0]

        rac = te.compute(
            (x, y, z), lambda i, j, k: te.div(env_prime[i, j, k], env[i, j, k])
        )
        rac_clipped = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                rac[i, j, k] > -1,
                te.if_then_else(
                    rac[i, j, k] < 1,
                    rac[i, j, k],
                    1,
                ),
                -1,
            ),
        )
        return (rac_clipped,)


class AmplitudeAcceleration:
    def __init__(self, computation_context={}):
        self._rac = RelativeAmplitudeChange()
        self._first_derivative = FirstDerivative()
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
        rac = self._rac._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        result = self._first_derivative._set_computation_context(
            {"X": rac, "x": x, "y": y, "z": z}
        )["result"][0]

        return (result,)


class InstantaneousFrequency:
    def __init__(self, sample_rate=4, computation_context={}):
        self._sample_rate = sample_rate
        self._phase = InstantaneousPhaseRadians()
        self._first_derivative = FirstDerivative()
        self._unwrap = Unwrap()
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
        pi_const = te.const(2 * np.pi, dtype=X.dtype)
        fs_const = te.const(1000 / self._sample_rate, dtype=X.dtype)
        phase = self._phase._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        unwrap = self._unwrap._set_computation_context(
            {"X": phase, "x": x, "y": y, "z": z}
        )["result"][0]

        phase_prime = self._first_derivative._set_computation_context(
            {"X": unwrap, "x": x, "y": y, "z": z}
        )["result"][0]

        freq = te.compute(
            (x, y, z),
            lambda i, j, k: te.abs(te.div(phase_prime[i, j, k], pi_const) * fs_const),
            name=get_name("freq"),
        )

        return (freq,)


class InstantaneousBandwidth:
    def __init__(self, computation_context={}):
        self._rac = RelativeAmplitudeChange()
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
        pi_const = te.const(2 * np.pi, X.dtype)
        rac = self._rac._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        result = te.compute(
            (x, y, z),
            lambda i, j, k: te.div(te.abs(rac[i, j, k]), pi_const),
        )

        return (result,)


class DominantFrequency:
    def __init__(self, computation_context={}):
        self._freq = InstantaneousFrequency()
        self._band = InstantaneousBandwidth()
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
        freq = self._freq._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        band = self._band._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        result = te.compute(
            (x, y, z),
            lambda i, j, k: te.sqrt(
                te.power(freq[i, j, k], 2) + te.power(band[i, j, k], 2)
            ),
        )

        return (result,)


class FrequencyChange:
    def __init__(self, computation_context={}):
        self._freq = InstantaneousFrequency()
        self._freq_prime = FirstDerivative()
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
        freq = self._freq._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        result = self._freq_prime._set_computation_context(
            {"X": freq, "x": x, "y": y, "z": z}
        )["result"][0]

        return (result,)


class Sweetness:
    def __init__(self, computation_context={}):
        self._env = Envelope()
        self._freq = InstantaneousFrequency()
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
        env = self._env._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        freq = self._freq._set_computation_context({"X": freq, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        freq_limit = te.compute(
            (x, y, z),
            lambda i, j, k: te.if_then_else(
                freq[i, j, k] < 5,
                5,
                freq[i, j, k],
            ),
        )

        result = te.compute(
            (x, y, z), lambda i, j, k: te.div(env[i, j, k], freq_limit[i, j, k])
        )

        return (result,)


class QualityFactor:
    def __init__(self, computation_context={}):
        self._rac = RelativeAmplitudeChange()
        self._freq = InstantaneousFrequency()
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
        pi_const = te.const(2 * np.pi, X.dtype)
        rac = self._rac._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        freq = self._freq._set_computation_context({"X": freq, "x": x, "y": y, "z": z})[
            "result"
        ][0]

        result = te.compute(
            (x, y, z), lambda i, j, k: te.div(pi_const * freq[i, j, k], rac[i, j, k])
        )

        return (result,)


# class ResponsePhase:
#     def __init__(self, computation_context={}):
#         self._envelope = Envelope()
#         self._phase = InstantaneousPhase()
#         self._local_events = LocalEvents()
#         self._cumsum = CumSum(axis=2)
#         self._operation = ResponseOperation()
#         self._set_computation_context(computation_context)

#     def _set_computation_context(self, computation_context):
#         x = computation_context.get("x", 1)
#         y = computation_context.get("y", 1)
#         z = computation_context.get("z", 64)
#         X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

#         result = self._computation_kernel(
#             X,
#             x,
#             y,
#             z,
#         )

#         self.computation_context = {
#             "result": result,
#             "input": (X,),
#         }
#         return self.computation_context

#     def _computation_kernel(self, X, x, y, z):
#         env = self._envelope._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
#             "result"
#         ][0]
#         phase = self._phase._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
#             "result"
#         ][0]
#         troughs = self._local_events._set_computation_context(
#             {"X": env, "x": x, "y": y, "z": z}
#         )["result"][0]
#         troughs = self._cumsum._set_computation_context(
#             {"X": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         result = self._operation._set_computation_context(
#             {"X1": env, "X2": phase, "X3": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         return (result,)

# class ResponseFrequency:
#     def __init__(self, computation_context={}):
#         self._envelope = Envelope()
#         self._freq = InstantaneousFrequency()
#         self._local_events = LocalEvents()
#         self._cumsum = CumSum(axis=2)
#         self._operation = ResponseOperation()
#         self._set_computation_context(computation_context)

#     def _set_computation_context(self, computation_context):
#         x = computation_context.get("x", 1)
#         y = computation_context.get("y", 1)
#         z = computation_context.get("z", 64)
#         X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

#         result = self._computation_kernel(
#             X,
#             x,
#             y,
#             z,
#         )

#         self.computation_context = {
#             "result": result,
#             "input": (X,),
#         }
#         return self.computation_context

#     def _computation_kernel(self, X, x, y, z):
#         env = self._envelope._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
#             "result"
#         ][0]
#         freq = self._freq._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
#             "result"
#         ][0]
#         troughs = self._local_events._set_computation_context(
#             {"X": env, "x": x, "y": y, "z": z}
#         )["result"][0]
#         troughs = self._cumsum._set_computation_context(
#             {"X": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         result = self._operation._set_computation_context(
#             {"X1": env, "X2": freq, "X3": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         return (result,)

# class ResponseAmplitude:
#     def __init__(self, computation_context={}):
#         self._envelope = Envelope()
#         self._local_events = LocalEvents()
#         self._cumsum = CumSum(axis=2)
#         self._operation = ResponseOperation()
#         self._set_computation_context(computation_context)

#     def _set_computation_context(self, computation_context):
#         x = computation_context.get("x", 1)
#         y = computation_context.get("y", 1)
#         z = computation_context.get("z", 64)
#         X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

#         result = self._computation_kernel(
#             X,
#             x,
#             y,
#             z,
#         )

#         self.computation_context = {
#             "result": result,
#             "input": (X,),
#         }
#         return self.computation_context

#     def _computation_kernel(self, X, x, y, z):
#         env = self._envelope._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
#             "result"
#         ][0]
#         troughs = self._local_events._set_computation_context(
#             {"X": env, "x": x, "y": y, "z": z}
#         )["result"][0]
#         troughs = self._cumsum._set_computation_context(
#             {"X": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         result = self._operation._set_computation_context(
#             {"X1": env, "X2": X, "X3": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         return (result,)

# class ApparentPolarity:
#     def __init__(self, computation_context={}):
#         self._envelope = Envelope()
#         self._local_events = LocalEvents()
#         self._cumsum = CumSum(axis=2)
#         self._operation = ResponseOperation(op_type="polarity")
#         self._set_computation_context(computation_context)

#     def _set_computation_context(self, computation_context):
#         x = computation_context.get("x", 1)
#         y = computation_context.get("y", 1)
#         z = computation_context.get("z", 64)
#         X = computation_context.get("X", te.placeholder((x, y, z), name=get_name("X")))

#         result = self._computation_kernel(
#             X,
#             x,
#             y,
#             z,
#         )

#         self.computation_context = {
#             "result": result,
#             "input": (X,),
#         }
#         return self.computation_context

#     def _computation_kernel(self, X, x, y, z):
#         env = self._envelope._set_computation_context({"X": X, "x": x, "y": y, "z": z})[
#             "result"
#         ][0]
#         troughs = self._local_events._set_computation_context(
#             {"X": env, "x": x, "y": y, "z": z}
#         )["result"][0]
#         troughs = self._cumsum._set_computation_context(
#             {"X": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         result = self._operation._set_computation_context(
#             {"X1": env, "X2": X, "X3": troughs, "x": x, "y": y, "z": z}
#         )["result"][0]

#         return (result,)


# class ResponseOperation:
#     def __init__(self, computation_context={}, op_type = "response"):
#         self._type = op_type
#         self._set_computation_context(computation_context)

#     def _set_computation_context(self, computation_context):
#         x = computation_context.get("x", 1)
#         y = computation_context.get("y", 1)
#         z = computation_context.get("z", 64)
#         X1 = computation_context.get(
#             "X1", te.placeholder((x, y, z), name=get_name("X1"))
#         )
#         X2 = computation_context.get(
#             "X2", te.placeholder((x, y, z), name=get_name("X2"))
#         )
#         X3 = computation_context.get(
#             "X3", te.placeholder((x, y, z), name=get_name("X3"))
#         )

#         result = self._computation_kernel(
#             X1,
#             X2,
#             X3,
#             x,
#             y,
#             z,
#         )

#         self.computation_context = {
#             "result": result,
#             "input": (X1, X2, X3),
#         }
#         return self.computation_context

#     def _computation_kernel(self, X1, X2, X3, x, y, z):
#         # Forward swap finding peaks
#         peaks_state = te.placeholder((x,y,z), name="peak_state")
#         peaks_init = te.compute((x,y,1), lambda i, j, _: 0, name="peaks_init")
#         peaks_update = te.compute(
#             (x,y,z),
#             lambda i, j, k: te.if_then_else(
#                 X3[i, j, k] == X3[i, j, k-1],
#                 te.if_then_else(
#                     X1[i, j, k] > X1[i, j, peaks_state[i, j, k-1]],
#                     k,
#                     peaks_state[i, j, k-1],
#                 ),
#                 k
#             ),
#             name="peaks_update"
#         )
#         peaks_scan = te.scan(peaks_init, peaks_update, peaks_state, inputs=[X1, X3], name="peaks_scan")

#         # Backward swap adjusting peak indexes
#         peaks_state = te.placeholder((x,y,z), name="peak_state_2")
#         peaks_init = te.compute((x,y,1), lambda i, j, _: peaks_scan[i, j, z-1], name="peaks_init_2")
#         peaks_update = te.compute(
#             (x,y,z),
#             lambda i, j, k: te.if_then_else(
#                 X3[i, j, z-k-1] == X3[i, j, z-k],
#                 peaks_state[i, j, k-1],
#                 peaks_scan[i, j, z-k-1]
#             ),
#             name="peaks_update_2"
#         )
#         peaks_scan = te.scan(peaks_init, peaks_update, peaks_state, inputs=[peaks_scan, X3], name="peaks_scan_2")
#         if self._type == "response":
#             result = te.compute(
#                 (x, y, z),
#                 lambda i, j, k: X2[i, j, peaks_scan[i, j, z-k+1]],
#                 name="operation_result"
#             )
#         else:
#             result = te.compute(
#                 (x, y, z),
#                 lambda i, j, k: te.if_then_else(
#                     X2[i, j, peaks_scan[i, j, z-k+1]] == 0,
#                     0,
#                     te.if_then_else(
#                         X2[i, j, peaks_scan[i, j, z-k+1]] > 0,
#                         X1[i, j, peaks_scan[i, j, z-k+1]],
#                         -X1[i, j, peaks_scan[i, j, z-k+1]],
#                     )
#                 ),
#                 name="operation_result"
#             )
#         return (result,)
