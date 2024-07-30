import argparse
import csv
import os
import timeit
import traceback
import multiprocessing
from multiprocessing import Process
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from jax_operators.operator_generic import JAXOperator
from utils import extract_data, weights

attrs = [
    # "hilbert",
    # "envelope",
    # "inst-phase",
    # "cos-inst-phase",
    # "relative-amplitude-change",
    # "amplitude-acceleration",
    # "inst-frequency",
    # "inst-bandwidth",
    # "dominant-frequency",
    # "frequency-change",
    # "sweetness",
    # "quality-factor",
    # "response-phase",
    # "response-frequency",
    # "response-amplitude",
    # "apparent-polarity",
    # "convolve1d",
    # "correlate1d",
    # "convolve2d",
    # "correlate2d",
    # "convolve3d",
    # "correlate3d",
    "glcm-asm",
    "glcm-contrast",
    "glcm-correlation",
    "glcm-variance",
    # "glcm-energy",
    # "glcm-entropy",
    "glcm-mean",
    "glcm-std",
    "glcm-dissimilarity",
    # "glcm-homogeneity",
]


def run_attr_op(args, name):
    data_id = int(os.path.basename(args.dataset).split(".")[0])
    second_dataset = os.path.join(os.path.dirname(args.dataset), f"{data_id%5 + 1}.npy")
    try:
        data = np.load(args.dataset).astype(args.dtype)
        data = extract_data(data, name)
        data = jax.device_put(data, device=jax.devices(args.arch)[0])
        op = JAXOperator(name)
        if "correlate" in name or "convolve" in name:
            weight = weights[name[-2:]].astype(args.dtype)
            weight = jax.device_put(weight, device=jax.devices(args.arch)[0])
            execution_times = timeit.repeat(
                f"op._transform_{args.arch}(data, weight)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
            del weight
        else:
            execution_times = timeit.repeat(
                f"op._transform_{args.arch}(data)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
        del data
        data_2 = np.load(second_dataset).astype(args.dtype)
        data_2 = extract_data(data_2, name)
        data_2 = jax.device_put(data_2, device=jax.devices(args.arch)[0])
        if "correlate" in name or "convolve" in name:
            weight_2 = weights[name[-2:]].astype(args.dtype)
            weight_2 = jax.device_put(weight_2, device=jax.devices(args.arch)[0])
            execution_times_2 = timeit.repeat(
                f"op._transform_{args.arch}(data_2, weight_2)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
        else:
            execution_times_2 = timeit.repeat(
                f"op._transform_{args.arch}(data_2)",
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
                f"jax_{args.arch}",
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
