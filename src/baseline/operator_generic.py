try:
    import cupy as cp
except:
    pass

from .complex_trace import (
    Hilbert,
    Envelope,
    InstantaneousPhase,
    CosineInstantaneousPhase,
    RelativeAmplitudeChange,
    AmplitudeAcceleration,
    InstantaneousFrequency,
    InstantaneousBandwidth,
    DominantFrequency,
    FrequencyChange,
    Sweetness,
    QualityFactor,
    ResponsePhase,
    ResponseFrequency,
    ResponseAmplitude,
    ApparentPolarity,
)

from .signal import (
    FFT,
    Convolve1D,
    Correlate1D,
    Convolve2D,
    Correlate2D,
    Convolve3D,
    Correlate3D,
)

from .texture import (
    GLCMASM,
    GLCMContrast,
    GLCMCorrelation,
    GLCMDissimilarity,
    GLCMEnergy,
    GLCMEntropy,
    GLCMHomogeneity,
    GLCMMean,
    GLCMStandardDeviation,
    GLCMVariance,
)

operators = {
    "fft": FFT, 
    "hilbert": Hilbert,
    "envelope": Envelope,
    "inst-phase": InstantaneousPhase,
    "cos-inst-phase": CosineInstantaneousPhase,
    "relative-amplitude-change": RelativeAmplitudeChange,
    "amplitude-acceleration": AmplitudeAcceleration,
    "inst-frequency": InstantaneousFrequency,
    "inst-bandwidth": InstantaneousBandwidth,
    "dominant-frequency": DominantFrequency,
    "frequency-change": FrequencyChange,
    "sweetness": Sweetness,
    "quality-factor": QualityFactor,
    "response-phase": ResponsePhase,
    "response-frequency": ResponseFrequency,
    "response-amplitude": ResponseAmplitude,
    "apparent-polarity": ApparentPolarity,
    "convolve1d": Convolve1D,
    "correlate1d": Correlate1D,
    "convolve2d": Convolve2D,
    "correlate2d": Correlate2D,
    "convolve3d": Convolve3D,
    "correlate3d": Correlate3D,
    "glcm-asm": GLCMASM,
    "glcm-contrast": GLCMContrast,
    "glcm-correlation": GLCMCorrelation,
    "glcm-dissimilarity": GLCMDissimilarity,
    "glcm-energy": GLCMEnergy,
    "glcm-entropy": GLCMEntropy,
    "glcm-homogeneity": GLCMHomogeneity,
    "glcm-mean": GLCMMean,
    "glcm-std": GLCMStandardDeviation,
    "glcm-variance": GLCMVariance,
}


class BaselineOperator:
    def __init__(self, operator, **kwargs):
        self.op = operators[operator](**kwargs)


    def _transform_cpu(self, *args):
        return self.op._transform_cpu(*args)

    def _transform_gpu(self, *args):
        ret = self.op._transform_gpu(*args)
        cp.cuda.runtime.deviceSynchronize()
        return ret
