import argparse
import json
import os
from glob import glob
from pathlib import Path

import tvm
import numpy as np

from tvm_te_operators.operator_generic import TVMOperator
from dasf_seismic.attributes.complex_trace import *
from dasf_seismic.attributes.texture import *
from baseline.signal import (
    Convolve1D,
    Convolve2D,
    Convolve3D,
    Correlate1D,
    Correlate2D,
    Correlate3D,
)
from utils import extract_data, extract_data_tvm, weights, weights_tvm

operators = {
    # "envelope": Envelope,
    # "inst-phase": InstantaneousPhase,
    # "cos-inst-phase": CosineInstantaneousPhase,
    # "relative-amplitude-change": RelativeAmplitudeChange,
    # "amplitude-acceleration": AmplitudeAcceleration,
    # "inst-frequency": InstantaneousFrequency,
    # "inst-bandwidth": InstantaneousBandwidth,
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
    # "glcm-asm": GLCMASM,
    # "glcm-contrast": GLCMContrast,
    # "glcm-correlation": GLCMCorrelation,
    # "glcm-variance": GLCMVariance,
    # "glcm-energy": GLCMEnergy,
    # "glcm-entropy": GLCMEntropy,
    # "glcm-mean": GLCMMean,
    # "glcm-std": GLCMStandardDeviation,
    # "glcm-dissimilarity": GLCMDissimilarity,
    # "glcm-homogeneity": GLCMHomogeneity
}


def header_arch(arch, filepath):
    with open(filepath, "a") as f:
        f.write(f"{'#'*500}\n#{arch.center(498)}#\n{'#'*500}\n")


def header_op(op, filepath):
    with open(filepath, "a") as f:
        f.write(f"{op.center(500, '+')}\n")


def validate(args):
    with open(args.index, "r") as f:
        scope = json.load(f)
        exp_id = (
            args.experiment_id
            if args.experiment_id is not None
            else list(scope.keys())[-1]
        )
        scope = scope[exp_id]
    dtype = scope["dtype"]
    if args.build:
        build_id = args.build
    else:
        with open(
            os.path.join(os.getcwd(), "experiments", "modules", exp_id, "index.json")
        ) as f:
            index_build = json.load(f)
            build_id = int(list(index_build.keys())[-1])
    obj_files = {
        "CPU": glob(
            os.path.join(
                os.getcwd(),
                "experiments",
                "modules",
                exp_id,
                f"Build{build_id:02d}",
                "*_cpu.so",
            )
        ),
        "GPU": glob(
            os.path.join(
                os.getcwd(),
                "experiments",
                "modules",
                exp_id,
                f"Build{build_id:02d}",
                "*_gpu.so",
            )
        ),
    }

    with open(args.file, "w") as f:
        f.write(f"EXP ID: {exp_id}  || BUILD ID: {build_id}\n")
    dataset_base = os.path.join("data", args.dataset, scope["data_size"])
    if args.cpu and obj_files["CPU"] != []:
        dev = tvm.cpu()
        header_arch("CPU", args.file)
        for module in obj_files["CPU"]:
            op = os.path.basename(module).replace("_cpu.so", "")
            if not (op in operators):
                continue
            op_tvm = TVMOperator(module, dev)
            op_base = operators[op]()
            header_op(op, args.file)
            result = []
            for dataset in sorted(glob(os.path.join(dataset_base, "*"))):
                data = np.load(dataset).astype(dtype)
                if "convolve" in op or "correlate" in op:
                    data_tvm = extract_data_tvm(data, op)
                    data = extract_data(data, op)
                    weight_tvm = weights_tvm[op[-2:]].astype(dtype)
                    weight = weights[op[-2:]].astype(dtype)
                    data_tvm = tvm.nd.array(data_tvm, device=dev)
                    weight_tvm = tvm.nd.array(weight_tvm, device=dev)
                    out_tvm = tvm.nd.empty(
                        data_tvm.shape, dtype=data_tvm.dtype, device=dev
                    )
                    op_tvm.transform(data_tvm, weight_tvm, out_tvm)
                    res_base = op_base._transform_cpu(data, weight)
                else:
                    data_tvm = tvm.nd.array(data, device=dev)
                    out_tvm = tvm.nd.empty(
                        data_tvm.shape, dtype=data_tvm.dtype, device=dev
                    )
                    op_tvm.transform(data_tvm, out_tvm)
                    res_base = op_base._transform_cpu(data)
                np.save("tvm_c.npy", out_tvm.numpy())
                np.save("cpu.npy", res_base)
                err = np.abs(out_tvm.numpy() - res_base)
                err_rel = np.abs(err / res_base)
                result.append((err.mean(), err.std(), err.max(), err_rel.max()))
            with open(args.file, "a") as f:
                result_str = ["|".join(list(map(str, r))).center(90) for r in result]
                f.write(
                    f"{dataset_base.center(30)} {dtype.center(9)} {' '.join(result_str)}\n"
                )
    if args.gpu and obj_files["GPU"] != []:
        import cupy as cp

        dev = tvm.cuda()
        header_arch("GPU", args.file)
        for module in obj_files["GPU"]:
            op = os.path.basename(module).replace("_gpu.so", "")
            if not (op in operators):
                continue
            op_tvm = TVMOperator(module, dev)
            op_base = operators[op]()
            header_op(op, args.file)
            result = []
            for dataset in sorted(glob(os.path.join(dataset_base, "*"))):
                data = np.load(dataset).astype(dtype)
                if "convolve" in op or "correlate" in op:
                    data_tvm = extract_data_tvm(data, op)
                    data = extract_data(data, op)
                    weight_tvm = weights_tvm[op[-2:]].astype(dtype)
                    weight = weights[op[-2:]].astype(dtype)
                    data_tvm = tvm.nd.array(data_tvm, device=dev)
                    weight_tvm = tvm.nd.array(weight_tvm, device=dev)
                    data = cp.asarray(data)
                    weight = cp.asarray(weight)
                    out_tvm = tvm.nd.empty(
                        data_tvm.shape, dtype=data_tvm.dtype, device=dev
                    )
                    op_tvm.transform(data_tvm, weight_tvm, out_tvm)
                    res_base = op_base._transform_gpu(data, weight)
                else:
                    data_tvm = tvm.nd.array(data, device=dev)
                    out_tvm = tvm.nd.empty(
                        data_tvm.shape, dtype=data_tvm.dtype, device=dev
                    )
                    data = cp.asarray(data)
                    op_tvm.transform(data_tvm, out_tvm)
                    res_base = op_base._transform_gpu(data)
                np.save("tvm_g.npy", out_tvm.numpy())
                np.save("gpu.npy", res_base.get())
                err = np.abs(out_tvm.numpy() - res_base.get())
                err_rel = np.abs(err / res_base.get())
                result.append((err.mean(), err.std(), err.max(), err_rel.max()))
            with open(args.file, "a") as f:
                result_str = ["|".join(list(map(str, r))).center(90) for r in result]
                f.write(
                    f"{dataset_base.center(30)} {dtype.center(9)} {' '.join(result_str)}\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--index",
        help="Experiment Index JSON file path",
        type=Path,
        default=os.path.join("experiments", "experiment_index.json"),
    )

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

    parser.add_argument(
        "-e",
        "--experiment-id",
        help="Experiment ID to use, if not set defaults to last on the file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-b",
        "--build",
        help="Build ID",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    validate(args)
