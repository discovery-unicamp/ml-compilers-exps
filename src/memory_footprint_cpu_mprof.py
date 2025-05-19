import argparse
import os
import csv
from pathlib import Path

from memory_profiler import memory_usage
import numpy as np


def measure_tvm(attribute, data_path, dtype, module_path, **kwargs):
    import tvm
    from tvm_te_operators.operator_generic import TVMOperator

    dev = tvm.cpu()
    op = TVMOperator(module_path, dev)
    op_2 = TVMOperator(module_path, dev)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data_tvm, weights_tvm
        data = np.load(data_path).astype(dtype)
        def run():
            data = extract_data_tvm(data, attribute)
            data_tvm = tvm.nd.array(data, device=dev)
            out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            weight_tvm = tvm.nd.array(weights_tvm[attribute[-2:]].astype(dtype), device=dev)
            op.transform(data_tvm, weight_tvm, out_tvm)

    elif attribute in ["fft", "hilbert"]:
        def run():
            data = np.load(data_path).astype(dtype)
            data_tvm = tvm.nd.array(data, device=dev)
            out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            out_tvm_2 = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            op.transform(data_tvm, out_tvm, out_tvm_2)


    else:
        def run():
            data = np.load(data_path).astype(dtype)
            data_tvm = tvm.nd.array(data, device=dev)
            out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            op.transform(data_tvm, out_tvm)

    mem_trace = memory_usage((run,))

    mem = max(mem_trace)


    return mem

def measure_torch(attribute, data_path, dtype, compile, **kwargs):
    import torch
    from torch_operators.operator_generic import TorchOperator

    op = TorchOperator(attribute, compile=compile)
    dev = torch.device("cpu")
    method = getattr(op, "_transform_cpu") if compile else getattr(op, "_nocompile_cpu")

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        def run():
            data = extract_data(data, attribute)
            data_torch = torch.from_numpy(data).to(dev)
            weight_torch = torch.from_numpy(weights[attribute[-2:]].astype(dtype)).to(dev)
            res = method(data_torch, weight_torch)
        
    else:
        def run():
            data = np.load(data_path).astype(dtype)
            data_torch = torch.from_numpy(data).to(dev)
            res = method(data_torch)
    
    mem_trace = memory_usage((run,))

    mem = max(mem_trace)

    return mem

def measure_jax(attribute, dtype, data_path, **kwargs):
    import jax
    jax.config.update("jax_enable_x64", True)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    from jax_operators.operator_generic import JAXOperator

    op = JAXOperator(attribute)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        def run():
            data = extract_data(data, attribute)
            data_jax = jax.device_put(data, device=jax.devices("cpu")[0])
            weight_jax = jax.device_put(weights[attribute[-2:]].astype(dtype), device=jax.devices("cpu")[0])
            res = op._transform_cpu(data_jax, weight_jax)
        
    else:
        def run():
            data = np.load(data_path).astype(dtype)
            data_jax = jax.device_put(data, device=jax.devices("cpu")[0])
            res = op._transform_cpu(data_jax)
    
    mem_trace = memory_usage((run,))

    mem = max(mem_trace)

    return mem


def measure_baseline(attribute, dtype, data_path, **kwargs):
    from baseline.operator_generic import BaselineOperator

    op = BaselineOperator(attribute)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        def run():
            data = extract_data(data, attribute)
            weight = weights[attribute[-2:]].astype(dtype)
            res = op._transform_cpu(data, weight)

    else:
        def run():
            data = np.load(data_path).astype(dtype)
            res = op._transform_cpu(data)
    
    mem_trace = memory_usage((run,))

    mem = max(mem_trace)
    
    return mem


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
    peak_mem = [peak_mem, 0]

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

    

