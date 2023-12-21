import argparse
import csv
import timeit
import traceback
from pathlib import Path
import cupy as cp

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
    "response-phase": ResponsePhase,
    "response-frequency": ResponseFrequency,
    "response-amplitude": ResponseAmplitude,
    "apparent-polarity": ApparentPolarity,
}


def run_exp(args):
    for name, attr in attrs.items():
        try:
            data = cp.load(args.dataset).astype(args.dtype)
            op = attr()
            execution_times = timeit.repeat(
                "op._transform_gpu(data)",
                repeat=args.repeat,
                number=1,
                globals=locals(),
            )
            del data
        except Exception as e:
            print(traceback(e))
            execution_times = [-1] * args.repeat
        with open(args.file, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([name, "gpu", 0, *execution_times])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
