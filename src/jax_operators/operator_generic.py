from jax import jit

from .complex_trace import (
    fft,
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
    convolve1d,
    correlate1d,
    convolve2d,
    correlate2d,
    convolve3d,
    correlate3d,
)

from .texture import (
    glcm_asm,
    glcm_contrast,
    glcm_correlation,
    glcm_dissimilarity,
    glcm_energy,
    glcm_entropy,
    glcm_homogeneity,
    glcm_mean,
    glcm_std,
    glcm_variance,
)

operators = {
    "fft": fft, 
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
    "convolve1d": convolve1d,
    "correlate1d": correlate1d,
    "convolve2d": convolve2d,
    "correlate2d": correlate2d,
    "convolve3d": convolve3d,
    "correlate3d": correlate3d,
    "glcm-asm": glcm_asm,
    "glcm-contrast": glcm_contrast,
    "glcm-correlation": glcm_correlation,
    "glcm-dissimilarity": glcm_dissimilarity,
    "glcm-energy": glcm_energy,
    "glcm-entropy": glcm_entropy,
    "glcm-homogeneity": glcm_homogeneity,
    "glcm-mean": glcm_mean,
    "glcm-std": glcm_std,
    "glcm-variance": glcm_variance,
}


class JAXOperator:
    def __init__(self, operator):
        function = operators[operator]
        self._cpu = jit(function, backend="cpu")
        self._gpu = jit(function, backend="cuda")

    def _transform_cpu(self, *args):
        return self._cpu(*args).block_until_ready()

    def _transform_gpu(self, *args):
        return self._gpu(*args).block_until_ready()
