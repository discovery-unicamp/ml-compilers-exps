from jax import jit

from .complex_trace import (
    hilbert,
    envelope,
    instantaneous_phase,
    cosine_instantaneous_phase,
    relative_amplitude_change,
    amplitude_acceleration,
    instantaneous_frequency,
    instantaneous_bandwidth,
    dominant_frequency,
    frequency_change,
    sweetness,
    quality_factor,
    response_phase,
    response_frequency,
    response_amplitude,
    apparent_polarity,
)

from .signal import (
    convolve1d_fft,
    convolve1d_direct,
    correlate1d_fft,
    correlate1d_direct,
    convolve2d_fft,
    convolve2d_direct,
    correlate2d_fft,
    correlate2d_direct
)

operators = {
    "hilbert": hilbert,
    "envelope": envelope,
    "inst-phase": instantaneous_phase,
    "cos-inst-phase": cosine_instantaneous_phase,
    "relative-amplitude-change": relative_amplitude_change,
    "amplitude-acceleration": amplitude_acceleration,
    "inst-frequency": instantaneous_frequency,
    "inst-bandwidth": instantaneous_bandwidth,
    "dominant-frequency": dominant_frequency,
    "frequency-change": frequency_change,
    "sweetness": sweetness,
    "quality-factor": quality_factor,
    "response-phase": response_phase,
    "response-frequency": response_frequency,
    "response-amplitude": response_amplitude,
    "apparent-polarity": apparent_polarity,
    "convolve1d_fft": convolve1d_fft,
    "convolve1d_direct": convolve1d_direct,
    "correlate1d_fft": correlate1d_fft,
    "correlate1d_direct": correlate1d_direct,
    "convolve2d_fft": convolve2d_fft,
    "convolve2d_direct": convolve2d_direct,
    "correlate2d_fft": correlate2d_fft,
    "correlate2d_direct": correlate2d_direct
}


class JAXOperator:
    def __init__(self, operator):
        function = operators[operator]
        self._cpu = jit(function, backend="cpu")
        self._gpu = jit(function, backend="cuda")

    def _transform_cpu(self, *args):
        return self._cpu(*args)

    def _transform_gpu(self, *args):
        return self._gpu(*args).block_until_ready()
