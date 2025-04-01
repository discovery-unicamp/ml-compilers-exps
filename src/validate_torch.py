import argparse
import os
import subprocess
from glob import glob

import numpy as np
import torch


from pathlib import Path
from torch_operators.operator_generic import TorchOperator
from baseline.complex_trace import *
from baseline.texture import *
from baseline.signal import (
    FFT,
    Convolve1D,
    Convolve2D,
    Convolve3D,
    Correlate1D,
    Correlate2D,
    Correlate3D,
)
from utils import extract_data, weights, check_attr_dataset_match

operators = {
    "fft": FFT,
    "envelope": Envelope,
    "inst-phase": InstantaneousPhase,
    "cos-inst-phase": CosineInstantaneousPhase,
    "relative-amplitude-change": RelativeAmplitudeChange,
    "amplitude-acceleration": AmplitudeAcceleration,
    # "inst-frequency": InstantaneousFrequency,
    "inst-bandwidth": InstantaneousBandwidth,
    # "dominant-frequency": DominantFrequency,
    # "frequency-change": FrequencyChange,
    # "sweetness": Sweetness,
    # "quality-factor": QualityFactor,
    # "response-phase": ResponsePhase,
    # "response-frequency": ResponseFrequency,
    # "response-amplitude": ResponseAmplitude,
    # "apparent-polarity": ApparentPolarity,
    "convolve1d": Convolve1D,
    "correlate1d": Correlate1D,
    "convolve2d": Convolve2D,
    "correlate2d": Correlate2D,
    "convolve3d": Convolve3D,
    "correlate3d": Correlate3D,
    "glcm-asm": GLCMASM,
    "glcm-contrast": GLCMContrast,
    # "glcm-correlation": GLCMCorrelation,
    "glcm-variance": GLCMVariance,
    "glcm-energy": GLCMEnergy,
    "glcm-entropy": GLCMEntropy,
    "glcm-mean": GLCMMean,
    "glcm-std": GLCMStandardDeviation,
    "glcm-dissimilarity": GLCMDissimilarity,
    "glcm-homogeneity": GLCMHomogeneity,
}


def get_git_revision_hash():
    subprocess.check_output(
        ["git", "config", "--global", "--add", "safe.directory", os.getcwd()]
    )  # bypass git repo ownership that leads to a problem when running inside a Docker container
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def header_arch(arch, filepath):
    with open(filepath, "a") as f:
        f.write(f"{'#'*500}\n#{arch.center(498)}#\n{'#'*500}\n")


def header_op(op, filepath):
    with open(filepath, "a") as f:
        f.write(f"{op.center(500, '+')}\n")


def validate(args):
    with open(args.file, "w") as f:
        f.write(f"GIT HASH: {get_git_revision_hash()}\n")
    dataset_base = os.path.join("data", args.dataset, "*")
    if args.cpu:
        header_arch("CPU", args.file)
        for op in operators.keys():
            op_base = operators[op]()
            header_op(op, args.file)
            for sh in sorted(glob(dataset_base)):
                if not check_attr_dataset_match(op, sh.split("/")[-1]):
                    continue
                results = {"float32": [], "float64": []}
                for dtype in results.keys():
                    op_torch = TorchOperator(op) # Torch operator needs to have compiler cache reset
                    for dataset in sorted(glob(os.path.join(sh, "*"))):
                        data = np.load(dataset).astype(dtype)
                        if "convolve" in op or "correlate" in op:
                            data = extract_data(data, op)
                            weight = weights[op[-2:]].astype(dtype)
                            data_torch = torch.from_numpy(data).to(torch.device("cpu"))
                            weight_torch = torch.from_numpy(weight).to(torch.device("cpu"))
                            res_torch = op_torch._transform_cpu(data_torch, weight_torch).numpy()
                            res_base = op_base._transform_cpu(data, weight)
                        else:
                            data_torch = torch.from_numpy(data).to(torch.device("cpu"))
                            res_torch = op_torch._transform_cpu(data_torch).numpy()
                            if "glcm" in op:  # GLCM baseline on CPU is too slow
                                import cupy as cp

                                res_base = op_base._transform_gpu(cp.array(data)).get()
                            else:
                                res_base = op_base._transform_cpu(data)
                        err = np.abs(res_torch - res_base)
                        res_base[res_base == 0] = 1 # err_rel computation retain abs err
                        err_rel = np.abs(err / res_base)
                        results[dtype].append(
                            (err.mean(), err.std(), err.max(), err_rel.max())
                        )
                with open(args.file, "a") as f:
                    for dtype, result in results.items():
                        result_str = [
                            "|".join(list(map(str, r))).center(90) for r in result
                        ]
                        f.write(
                            f"{sh.center(30)} {dtype.center(9)} {' '.join(result_str)}\n"
                        )
    if args.gpu:
        import cupy as cp

        header_arch("GPU", args.file)
        for op in operators.keys():
            op_torch = TorchOperator(op)
            op_base = operators[op]()
            header_op(op, args.file)
            for sh in sorted(glob(dataset_base)):
                if not check_attr_dataset_match(op, sh.split("/")[-1]):
                    continue
                results = {"float32": [], "float64": []}
                for dtype in results.keys():
                    op_torch = TorchOperator(op) # Torch operator needs to have compiler cache reset
                    for dataset in sorted(glob(os.path.join(sh, "*"))):
                        data = np.load(dataset).astype(dtype)
                        if "convolve" in op or "correlate" in op:
                            data = extract_data(data, op)
                            weight = weights[op[-2:]].astype(dtype)
                            data_torch = torch.from_numpy(data).to(torch.device("cuda"))
                            weight_torch = torch.from_numpy(weight).to(torch.device("cuda"))
                            data = cp.asarray(data)
                            weight = cp.asarray(weight)
                            res_torch = op_torch._transform_gpu(data_torch, weight_torch)
                            res_base = op_base._transform_gpu(data, weight)
                        else:
                            data_torch = torch.from_numpy(data).to(torch.device("cuda"))
                            data = cp.asarray(data)
                            res_torch = op_torch._transform_gpu(data_torch)
                            res_base = op_base._transform_gpu(data)
                        err = cp.abs(res_torch - res_base)
                        res_base[res_base == 0] = 1 # err_rel computation retain abs err
                        err_rel = cp.abs(err / res_base)
                        results[dtype].append(
                            (err.mean(), err.std(), err.max(), err_rel.max())
                        )
                with open(args.file, "a") as f:
                    for dtype, result in results.items():
                        result_str = [
                            "|".join(list(map(str, r))).center(90) for r in result
                        ]
                        f.write(
                            f"{sh.center(30)} {dtype.center(9)} {' '.join(result_str)}\n"
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        help="path to validation report",
        type=Path,
        required=True,
        default=None,
    )

    parser.add_argument(
        "-c",
        "--cpu",
        help="CPU will be used",
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help="GPU will be used",
        action="store_true",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="dataset to use",
        type=str,
        choices=["parihaka"],
        default="parihaka",
    )

    args = parser.parse_args()
    validate(args)
