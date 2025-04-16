import torch

from .complex_trace import (
    fft,
    hilbert,
    envelope,
    instantaneous_phase,
    cosine_instantaneous_phase,
    relative_amplitude_change,
    amplitude_acceleration,
    instantaneous_bandwidth,
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
    # "inst-frequency": instantaneous_frequency,
    "inst-bandwidth": instantaneous_bandwidth,
    # "dominant-frequency": dominant_frequency,
    # "frequency-change": frequency_change,
    # "sweetness": sweetness,
    # "quality-factor": quality_factor,
    # "response-phase": response_phase,
    # "response-frequency": response_frequency,
    # "response-amplitude": response_amplitude,
    # "apparent-polarity": apparent_polarity,
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


class TorchOperator:
    def __init__(self, operator, backend="inductor"):
        function = operators[operator]
        self.func = function
        torch.compiler.reset()
        self.op = torch.compile(function, backend=backend)

    def _transform_cpu(self, *args):
        res = self.op(*args)
        torch.cpu.synchronize()
        return res

    def _transform_gpu(self, *args):
        res = self.op(*args)
        torch.cuda.synchronize()
        return res
    
    def _nocompile_cpu(self, *args):
        res = self.func(*args)
        torch.cpu.synchronize()
        return res

    def _nocompile_gpu(self, *args):
        res = self.func(*args)
        torch.cuda.synchronize()
        return res
    

