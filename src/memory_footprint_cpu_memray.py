import argparse
import os
import csv
from pathlib import Path

import numpy as np
import memray
from memray._memray import compute_statistics


def measure_tvm(attribute, data_path, dtype, module_path, **kwargs):
    import tvm
    from tvm_te_operators.operator_generic import TVMOperator

    dev = tvm.cpu()
    op = TVMOperator(module_path, dev)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data_tvm, weights_tvm
        data = np.load(data_path).astype(dtype)
        with memray.Tracker("output_tvm.bin"):
            data = extract_data_tvm(data, attribute)
            data_tvm = tvm.nd.array(data, device=dev)
            out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            weight_tvm = tvm.nd.array(weights_tvm[attribute[-2:]].astype(dtype), device=dev)
            op.transform(data_tvm, weight_tvm, out_tvm)
        with memray.Tracker("output_tvm_2.bin"):
            op.transform(data_tvm, weight_tvm, out_tvm)

    elif attribute in ["fft", "hilbert"]:
        with memray.Tracker("output_tvm.bin"):
            data = np.load(data_path).astype(dtype)
            data_tvm = tvm.nd.array(data, device=dev)
            out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            out_tvm_2 = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            op.transform(data_tvm, out_tvm, out_tvm_2)

        with memray.Tracker("output_tvm_2.bin"):
            op.transform(data_tvm, out_tvm, out_tvm_2)

    else:
        with memray.Tracker("output_tvm.bin"):
            data = np.load(data_path).astype(dtype)
            data_tvm = tvm.nd.array(data, device=dev)
            out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            op.transform(data_tvm, out_tvm)

        with memray.Tracker("output_tvm_2.bin"):
            op.transform(data_tvm, out_tvm)
    
    stats = compute_statistics("output_tvm.bin")
    mem = stats.peak_memory_allocated/(1024**2)

    stats = compute_statistics("output_tvm_2.bin")
    mem_2 = stats.peak_memory_allocated/(1024**2)

    os.remove("output_tvm.bin")
    os.remove("output_tvm_2.bin")

    return [mem, mem_2]

def measure_torch(attribute, data_path, dtype, compile, **kwargs):
    import torch
    from torch_operators.operator_generic import TorchOperator

    op = TorchOperator(attribute, compile=compile)
    dev = torch.device("cpu")
    method = getattr(op, "_transform_cpu") if compile else getattr(op, "_nocompile_cpu")

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        with memray.Tracker("output_torch.bin"):
            data = extract_data(data, attribute)
            data_torch = torch.from_numpy(data).to(dev)
            weight_torch = torch.from_numpy(weights[attribute[-2:]].astype(dtype)).to(dev)
            res = method(data_torch, weight_torch)
        with memray.Tracker("output_torch_2.bin"):
            res = method(data_torch, weight_torch)
        
    else:
        with memray.Tracker("output_torch.bin"):
            data = np.load(data_path).astype(dtype)
            data_torch = torch.from_numpy(data).to(dev)
            res = method(data_torch)

        with memray.Tracker("output_torch_2.bin"):
            res_2 = method(data_torch)
    
    stats = compute_statistics("output_torch.bin")
    mem = stats.peak_memory_allocated/(1024**2)

    stats = compute_statistics("output_torch_2.bin")
    mem_2 = stats.peak_memory_allocated/(1024**2)

    os.remove("output_torch.bin")
    os.remove("output_torch_2.bin")

    return [mem, mem_2]

def measure_jax(attribute, dtype, data_path, **kwargs):
    import jax
    jax.config.update("jax_enable_x64", True)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    from jax_operators.operator_generic import JAXOperator

    op = JAXOperator(attribute)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        with memray.Tracker("output_jax.bin"):
            data = extract_data(data, attribute)
            data_jax = jax.device_put(data, device=jax.devices("cpu")[0])
            weight_jax = jax.device_put(weights[attribute[-2:]].astype(dtype), device=jax.devices("cpu")[0])
            res = op._transform_cpu(data_jax, weight_jax)
        with memray.Tracker("output_jax_2.bin"):
            res = op._transform_cpu(data_jax, weight_jax)
        
    else:
        with memray.Tracker("output_jax.bin"):
            data = np.load(data_path).astype(dtype)
            data_jax = jax.device_put(data, device=jax.devices("cpu")[0])
            res = op._transform_cpu(data_jax)

        with memray.Tracker("output_jax_2.bin"):
            res_2 = op._transform_cpu(data_jax)
    
    stats = compute_statistics("output_jax.bin")
    mem = stats.peak_memory_allocated/(1024**2)

    stats = compute_statistics("output_jax_2.bin")
    mem_2 = stats.peak_memory_allocated/(1024**2)

    os.remove("output_jax.bin")
    os.remove("output_jax_2.bin")

    return [mem, mem_2]


def measure_baseline(attribute, dtype, data_path, **kwargs):
    from baseline.operator_generic import BaselineOperator

    op = BaselineOperator(attribute)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        with memray.Tracker("output_baseline.bin"):
            data = extract_data(data, attribute)
            weight = weights[attribute[-2:]].astype(dtype)
            res = op._transform_cpu(data, weight)
        with memray.Tracker("output_baseline_2.bin"):
            res = op._transform_cpu(data, weight)
        
    else:
        with memray.Tracker("output_baseline.bin"):
            data = np.load(data_path).astype(dtype)
            res = op._transform_cpu(data)

        with memray.Tracker("output_baseline_2.bin"):
            res_2 = op._transform_cpu(data)
    
    stats = compute_statistics("output_baseline.bin")
    mem = stats.peak_memory_allocated/(1024**2)

    stats = compute_statistics("output_baseline_2.bin")
    mem_2 = stats.peak_memory_allocated/(1024**2)

    os.remove("output_baseline.bin")
    os.remove("output_baseline_2.bin")

    return [mem, mem_2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runtime",
        help="runtime to use",
        type=str,
        choices=["baseline_cpu", "tvm_cpu_1", "tvm_cpu_2", "jax_cpu", "torch_c_cpu", "torch_n_cpu"],
        required=True,
    )

    parser.add_argument("-f", "--file", help="CSV file path", type=Path, required=True)

    parser.add_argument(
        "-a",
        "--attribute",
        help="attribute to run",
        type=str,
        choices=[
            "fft",
            "envelope",
            "inst-phase",
            "cos-inst-phase",
            "relative-amplitude-change",
            "amplitude-acceleration",
            "inst-bandwidth",
            "convolve1d",
            "correlate1d",
            "convolve2d",
            "correlate2d",
            "convolve3d",
            "correlate3d",
            "glcm-asm",
            "glcm-contrast",
            "glcm-variance",
            "glcm-energy",
            "glcm-entropy",
            "glcm-mean",
            "glcm-std",
            "glcm-dissimilarity",
            "glcm-homogeneity",
        ]
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="NPY dataset file path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-i",
        "--dataset-id",
        help="dataset id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-b",
        "--base-path",
        help="path to module files",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-t",
        "--dtype",
        help="data type to use",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    args = parser.parse_args()

    measure_dict = {
        "baseline_cpu": (measure_baseline, {}),
        "tvm_cpu_1": (measure_tvm, {"module_path": os.path.join(args.base_path, "Build01", f"{args.attribute}_cpu.so")}),
        "tvm_cpu_2": (measure_tvm, {"module_path": os.path.join(args.base_path, "Build02", f"{args.attribute}_cpu.so")}),
        "jax_cpu": (measure_jax, {}),
        "torch_c_cpu": (measure_torch, {"compile": True}),
        "torch_n_cpu": (measure_torch, {"compile": False})
    }

    func, kwargs = measure_dict[args.runtime]
    peak_mem = func(
        attribute=args.attribute,
        dtype=args.dtype,
        data_path=args.dataset,
        **kwargs
    )

    with open(args.file, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                args.attribute,
                args.dataset_id,
                args.runtime,
                os.getenv("OMP_NUM_THREADS", 0),
                *peak_mem
            ]
        )

    

