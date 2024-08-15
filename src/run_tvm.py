import argparse
import csv
import os
import timeit
import traceback
import multiprocessing
from multiprocessing import Process
from pathlib import Path

import numpy as np
from tvm_te_operators.operator_generic import TVMOperator

import tvm

from utils import extract_data_tvm, weights_tvm

attrs = [
    "fft",
    # "hilbert",
    "envelope",
    "inst-phase",
    "cos-inst-phase",
    "relative-amplitude-change",
    "amplitude-acceleration",
    # "inst-frequency",
    "inst-bandwidth",
    # "dominant-frequency",
    # "frequency-change",
    # "sweetness",
    # "quality-factor",
    # "response-phase",
    # "response-frequency",
    # "response-amplitude",
    # "apparent-polarity",
    "convolve1d",
    # "correlate1d",
    "convolve2d",
    # "correlate2d",
    "convolve3d",
    # "correlate3d",
    "glcm-asm",
    "glcm-contrast",
    # "glcm-correlation",
    "glcm-variance",
    "glcm-energy",
    "glcm-entropy",
    "glcm-mean",
    "glcm-std",
    "glcm-dissimilarity",
    "glcm-homogeneity",
]


def run_attr_op(args, name):
    dev = tvm.cuda() if args.arch == "gpu" else tvm.cpu()
    module = os.path.join(args.base_path, f"{name}_{args.arch}.so")
    data_id = int(os.path.basename(args.dataset).split(".")[0])
    second_dataset = os.path.join(os.path.dirname(args.dataset), f"{data_id%5 + 1}.npy")
    if not os.path.isfile(module):
        return
    try:
        data = np.load(args.dataset).astype(args.dtype)
        data = extract_data_tvm(data, name)
        data_tvm = tvm.nd.array(data, device=dev)
        out_tvm = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
        op = TVMOperator(module, dev)
        if name in ["fft", "hilbert"]:
            out_tvm_2 = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            execution_times = timeit.repeat(
                "op.transform(data_tvm, out_tvm, out_tvm_2)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
            del out_tvm_2
        elif "convolve" in name or "correlate" in name:
            weight = weights_tvm[name[-2:]].astype(args.dtype)
            weight_tvm = tvm.nd.array(weight, device=dev)
            execution_times = timeit.repeat(
                "op.transform(data_tvm, weight_tvm, out_tvm)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
            del weight_tvm
        else:
            execution_times = timeit.repeat(
                "op.transform(data_tvm, out_tvm)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
        del data, out_tvm
        data_2 = np.load(second_dataset).astype(args.dtype)
        data_2 = extract_data_tvm(data_2, name)
        data_2_tvm = tvm.nd.array(data_2, device=dev)
        out_2_tvm = tvm.nd.empty(data_2_tvm.shape, dtype=data_2_tvm.dtype, device=dev)
        if name in ["fft", "hilbert"]:
            out_2_tvm_2 = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=dev)
            execution_times_2 = timeit.repeat(
                "op.transform(data_2_tvm, out_2_tvm, out_2_tvm_2)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
        elif "convolve" in name or "correlate" in name:
            weight = weights_tvm[name[-2:]].astype(args.dtype)
            weight_tvm = tvm.nd.array(weight, device=dev)
            execution_times_2 = timeit.repeat(
                "op.transform(data_2_tvm, weight_tvm, out_2_tvm)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
            del weight_tvm
        else:
            execution_times_2 = timeit.repeat(
                "op.transform(data_2_tvm, out_2_tvm)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
    except Exception as e:
        print(traceback(e))
        execution_times = [-1] * args.repeat
    with open(args.file, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                name,
                f"tvm_{args.arch}",
                os.getenv("OMP_NUM_THREADS", 0),
                *execution_times,
                *execution_times_2,
            ]
        )


def run_exp(args):
    multiprocessing.set_start_method("spawn", force=True)
    for name in attrs:
        p = Process(target=run_attr_op, args=(args, name))
        p.start()
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        help="device to use, CPU or GPU",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
    )

    parser.add_argument(
        "-b",
        "--base-path",
        help="path to module files",
        type=Path,
        required=True,
    )

    parser.add_argument("-f", "--file", help="CSV file path", type=Path, required=True)

    parser.add_argument(
        "-r",
        "--repeat",
        help="number of samples",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="NPY dataset file path",
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
    run_exp(args)
