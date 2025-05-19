import argparse
import os
import csv
import shutil
import subprocess
from pathlib import Path
import sqlite3

import numpy as np


def measure_tvm(attribute, data_path, dtype, module_path, **kwargs):
    import tvm
    from tvm_te_operators.operator_generic import TVMOperator

    dev = tvm.cuda()
    op = TVMOperator(module_path, dev)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data_tvm, weights_tvm
        data = np.load(data_path).astype(dtype)
        data = extract_data_tvm(data, attribute)
        data_tvm = tvm.nd.array(data, device=dev)
        out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
        weight_tvm = tvm.nd.array(weights_tvm[attribute[-2:]].astype(dtype), device=dev)
        op.transform(data_tvm, weight_tvm, out_tvm)

    elif attribute in ["fft", "hilbert"]:
        data = np.load(data_path).astype(dtype)
        data_tvm = tvm.nd.array(data, device=dev)
        out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
        out_tvm_2 = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
        op.transform(data_tvm, out_tvm, out_tvm_2)


    else:
        data = np.load(data_path).astype(dtype)
        data_tvm = tvm.nd.array(data, device=dev)
        out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
        op.transform(data_tvm, out_tvm)
    

    return

def measure_torch(attribute, data_path, dtype, compile, **kwargs):
    import torch
    from torch_operators.operator_generic import TorchOperator

    op = TorchOperator(attribute, compile=compile)
    dev = torch.device("cuda")
    method = getattr(op, "_transform_gpu") if compile else getattr(op, "_nocompile_gpu")

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        data = extract_data(data, attribute)
        data_torch = torch.from_numpy(data).to(dev)
        weight_torch = torch.from_numpy(weights[attribute[-2:]].astype(dtype)).to(dev)
        res = method(data_torch, weight_torch)
        
    else:
        data = np.load(data_path).astype(dtype)
        data_torch = torch.from_numpy(data).to(dev)
        res = method(data_torch)

    return

def measure_jax(attribute, dtype, data_path, **kwargs):
    import jax
    jax.config.update("jax_enable_x64", True)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    from jax_operators.operator_generic import JAXOperator

    op = JAXOperator(attribute)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = np.load(data_path).astype(dtype)
        data = extract_data(data, attribute)
        data_jax = jax.device_put(data, device=jax.devices("gpu")[0])
        weight_jax = jax.device_put(weights[attribute[-2:]].astype(dtype), device=jax.devices("gpu")[0])
        res = op._transform_gpu(data_jax, weight_jax)
        
    else:
        data = np.load(data_path).astype(dtype)
        data_jax = jax.device_put(data, device=jax.devices("gpu")[0])
        res = op._transform_gpu(data_jax)
    
    return


def measure_baseline(attribute, dtype, data_path, **kwargs):
    from baseline.operator_generic import BaselineOperator
    import cupy as cp

    op = BaselineOperator(attribute)
    data = np.load(data_path).astype(dtype)

    if "correlate" in attribute or "convolve" in attribute:
        from utils import extract_data, weights
        data = extract_data(data, attribute)
        weight = weights[attribute[-2:]].astype(dtype)
        data = cp.asarray(data)
        weight = cp.asarray(weight)
        res = op._transform_gpu(data, weight)
        
    else:
        data = cp.asarray(data)
        res = op._transform_gpu(data)

    return

def run_with_nsys(args):
    process = subprocess.run(
        [
            f"./scripts/memory_footprint_gpu.sh",
            args.file,
            args.attribute,
            args.runtime,
            args.dtype,
            "0",
            args.base_path,
            str(args.dataset_id),
            args.dataset,
            "true"
        ],
        capture_output=True,
    )


    alloc_list = [
        [0]
        for i in range(8)
    ]
    con = sqlite3.connect("gpu_mem_trace.sqlite")
    cur = con.cursor()
    ex = cur.execute("SELECT start,bytes,memKind,memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start;")
    allocations = ex.fetchall()

    for alloc in allocations:
        mult = -1 if alloc[3] else 1
        alloc_list[alloc[2]].append(alloc[1]*mult)

    peak_mem = [
        np.max(np.cumsum(trace))
        for trace in alloc_list
    ]

    return peak_mem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runtime",
        help="runtime to use",
        type=str,
        choices=["baseline_gpu", "tvm_gpu", "jax_gpu", "torch_c_gpu", "torch_n_gpu"],
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
        "-n",
        "--nsys",
        help="using Nsight Systems",
        action="store_true",
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

    if args.nsys:
        measure_dict = {
            "baseline_gpu": (measure_baseline, {}),
            "tvm_gpu": (measure_tvm, {"module_path": os.path.join(args.base_path, "Build03", f"{args.attribute}_gpu.so")}),
            "jax_gpu": (measure_jax, {}),
            "torch_c_gpu": (measure_torch, {"compile": True}),
            "torch_n_gpu": (measure_torch, {"compile": False})
        }

        func, kwargs = measure_dict[args.runtime]

        func(
            attribute=args.attribute,
            dtype=args.dtype,
            data_path=args.dataset,
            **kwargs
        )
    else:
        peak_mem = run_with_nsys(args=args)
        with open(args.file, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    args.attribute,
                    args.dataset_id,
                    args.runtime,
                    os.getenv("OMP_NUM_THREADS", 0),
                    *peak_mem,
                ]
            )

    

