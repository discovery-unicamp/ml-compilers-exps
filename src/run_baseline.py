import argparse
import csv
import os
import timeit
import traceback
from pathlib import Path
import multiprocessing
from multiprocessing import Process

import numpy as np

try:
    import cupy as cp
except:
    pass

from dasf_seismic.attributes.complex_trace import (
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

attrs = {
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
    # "response-phase": ResponsePhase,
    # "response-frequency": ResponseFrequency,
    # "response-amplitude": ResponseAmplitude,
    # "apparent-polarity": ApparentPolarity,
}


def run_attr_op(args, name):
    arch = "gpu" if args.baseline == "cupy" else "cpu"
    data_id = int(os.path.basename(args.dataset).split(".")[0])
    second_dataset = os.path.join(os.path.dirname(args.dataset), f"{data_id%5 + 1}.npy")
    try:
        data = (
            np.load(args.dataset).astype(args.dtype)
            if arch == "cpu"
            else cp.load(args.dataset).astype(args.dtype)
        )
        op = attrs[name]()
        execution_times = timeit.repeat(
            f"op._transform_{arch}(data)",
            repeat=args.repeat,
            number=1,
            globals=locals(),
        )
        del data
        data_2 = (
            np.load(second_dataset).astype(args.dtype)
            if arch == "cpu"
            else cp.load(second_dataset).astype(args.dtype)
        )
        execution_times_2 = timeit.repeat(
            f"op._transform_{arch}(data_2)",
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
                args.baseline,
                os.getenv("OMP_NUM_THREADS", 0),
                *execution_times,
                *execution_times_2,
            ]
        )


def run_exp(args):
    multiprocessing.set_start_method("spawn", force=True)
    for name, attr in attrs.items():
        p = Process(target=run_attr_op, args=(args, name))
        p.start()
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--baseline",
        help="baseline being run",
        type=str,
        choices=["intel_conda", "amd", "amd_gcc", "openblas", "default", "cupy"],
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
