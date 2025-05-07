# Implementation based on https://github.com/discovery-unicamp/dasf-seismic/blob/main/dasf_seismic/attributes/texture.py
import numpy as np

try:
    import cupy as cp
    from glcm_cupy import glcm as glcm_gpu
    from glcm_cupy import Features as glcm_features
    from glcm_cupy import Direction as glcm_direction
except ImportError:
    pass

from skimage.feature import graycomatrix, graycoprops


def glcm_mean(glcm):
    I = np.ogrid[0 : len(glcm)]
    return np.sum(I * glcm, axis=(0, 1))


def glcm_var(glcm):
    mean = glcm_mean(glcm)
    I = np.ogrid[0 : len(glcm)]
    return np.sum(glcm * ((I - mean) ** 2).T, axis=(0, 1))


def glcm_std(glcm):
    var = glcm_var(glcm)
    return np.sqrt(var)


def glcm_entropy(glcm):
    ln = -np.log(glcm, where=(glcm != 0), out=np.zeros_like(glcm))
    return np.sum(glcm * ln, axis=(0, 1))


def get_glcm_gpu_feature(glcm_type):
    if glcm_type == "contrast":
        return glcm_features.CONTRAST
    elif glcm_type == "dissimilarity":
        return glcm_features.DISSIMILARITY
    elif glcm_type == "homogeneity":
        return glcm_features.HOMOGENEITY
    elif glcm_type == "asm" or glcm_type == "energy":
        return glcm_features.ASM
    elif glcm_type == "mean":
        return glcm_features.MEAN
    elif glcm_type == "correlation":
        return glcm_features.CORRELATION
    elif glcm_type == "var" or glcm_type == "std":
        return glcm_features.VARIANCE
    elif glcm_type == "entropy":
        return glcm_features.ENTROPY
    else:
        raise NotImplementedError("GLCM type '%s' is not supported." % glcm_type)


def get_glcm_gpu_direction(direction):
    if direction == 0:
        return glcm_direction.EAST
    elif direction == np.pi / 4:
        return glcm_direction.SOUTH_EAST
    elif direction == np.pi / 2:
        return glcm_direction.SOUTH
    elif direction == 3 * np.pi / 2:
        return glcm_direction.SOUTH_WEST
    else:
        raise NotImplementedError(
            "GLCM direction angle '%s' is not supported." % direction
        )


def get_glcm_cpu_feature(glcm_type):
    if glcm_type == "asm":
        return "ASM"
    return glcm_type


class GLCMGeneric:
    def __init__(
        self,
        glcm_type,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
        preview=None,
    ):
        super().__init__()

        self._glcm_type = glcm_type
        self._levels = levels
        self._direction = direction
        self._distance = distance
        self._window = window
        self._preview = preview

    def _operation_cpu(
        self,
        block,
        glcm_type_block,
        levels_block,
        direction_block,
        distance_block,
        window,
        glb_mi,
        glb_ma,
    ):
        funcs = {
            "entropy": glcm_entropy,
            "std": glcm_std,
            "var": glcm_var,
            "mean": glcm_mean,
        }
        custom_func = funcs.get(glcm_type_block, None)

        assert len(window) == 2
        kh, kw = np.array(window) // 2
        new_atts = list()
        gl = ((block - glb_mi) / (glb_ma - glb_mi)) * (levels_block - 1)
        gl = gl.astype(int)
        gl = np.pad(gl, pad_width=((0, 0), (kh, kh), (kw, kw)), constant_values=0)
        (
            d,
            h,
            w,
        ) = gl.shape
        for k in range(d):
            new_att = np.zeros((h, w), dtype=np.float32)
            gl_block = gl[k]
            for i in range(h):
                for j in range(w):
                    # Windows needs to fit completely in image
                    if i < kh or j < kw:
                        continue
                    if i > (h - kh - 1) or j > (w - kw - 1):
                        continue

                    # Calculate GLCM on a kh x kw window, default is 7x7
                    glcm_window = gl_block[i - kh : i + kh + 1, j - kw : j + kw + 1]
                    glcm = graycomatrix(
                        glcm_window,
                        [distance_block],
                        [direction_block],
                        levels=levels_block,
                        symmetric=True,
                        normed=True,
                    )

                    # Calculate property and replace center pixel
                    if custom_func:
                        res = custom_func(glcm[:, :, 0, 0])
                    else:
                        res = graycoprops(glcm, glcm_type_block)
                    new_att[i, j] = res

            new_atts.append(new_att.astype(block.dtype))
        result = np.asarray(new_atts, dtype=block.dtype)
        return result[:, kh:-kh, kw:-kw]

    def _operation_gpu(
        self,
        block,
        glcm_type_block,
        levels_block,
        direction_block,
        distance_block,
        window,
        glb_mi,
        glb_ma,
    ):
        assert len(window) == 2
        kh, kw = np.array(window) // 2
        radius = max(kh, kw)
        step_size = distance_block
        padding = radius + step_size

        gl = ((block - glb_mi) / (glb_ma - glb_mi)) * (levels_block - 1)
        gl = gl
        gl_pad = cp.pad(
            gl,
            pad_width=((0, 0), (padding, padding), (padding, padding)),
            constant_values=0,
        )
        image_batch = gl_pad[:, :, :, cp.newaxis]
        g = glcm_gpu(
            image_batch,
            directions=[direction_block],
            features=[glcm_type_block],
            step_size=step_size,
            radius=radius,
            bin_from=levels_block,
            bin_to=levels_block,
            normalized_features=False,
            skip_border=True,
            verbose=False,
        )
        return cp.asarray(
            g[..., glcm_type_block].squeeze(axis=3),
        )

    def _transform_gpu(self, X, glb_mi=None, glb_ma=None):
        mi = cp.min(X) if glb_mi is None else glb_mi
        ma = cp.max(X) if glb_ma is None else glb_ma

        glcm_type = get_glcm_gpu_feature(self._glcm_type)
        direction = get_glcm_gpu_direction(self._direction)

        X = self._operation_gpu(
            X, glcm_type, self._levels, direction, self._distance, self._window, mi, ma
        )

        if self._glcm_type in ["std", "energy"]:
            X = cp.sqrt(X)

        return X

    def _transform_cpu(self, X, glb_mi=None, glb_ma=None):
        mi = np.min(X) if glb_mi is None else glb_mi
        ma = np.max(X) if glb_ma is None else glb_ma

        glcm_type = get_glcm_cpu_feature(self._glcm_type)

        return self._operation_cpu(
            X,
            glcm_type,
            self._levels,
            self._direction,
            self._distance,
            self._window,
            mi,
            ma,
        )


class GLCMContrast(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="contrast",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMDissimilarity(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="dissimilarity",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMASM(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="asm",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMMean(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="mean",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMCorrelation(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="correlation",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMHomogeneity(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="homogeneity",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMVariance(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="var",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMEntropy(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="entropy",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMStandardDeviation(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="std",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )


class GLCMEnergy(GLCMGeneric):
    def __init__(
        self,
        levels=16,
        direction=np.pi / 2,
        distance=1,
        window=(7, 7),
    ):
        super().__init__(
            glcm_type="energy",
            levels=levels,
            direction=direction,
            distance=distance,
            window=window,
        )
