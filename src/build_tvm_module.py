import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np

import tvm
from tvm import te, auto_scheduler

from tvm_te_operators.complex_trace import (
    FFT,
    IFFT,
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
    # ResponsePhase,
    # ResponseFrequency,
    # ResponseAmplitude,
    # ApparentPolarity,
)


from tvm_te_operators.texture.glcm import (
    GLCMBase,
    GLCMASM,
    GLCMContrast,
    GLCMCorrelation,
    GLCMVariance,
    GLCMEnergy,
    GLCMEntropy,
    GLCMMean,
    GLCMStandardDeviation,
    GLCMDissimilarity,
    GLCMHomogeneity,
)

from tvm_te_operators.signal.convolution import (
    Convolution1D,
    Correlation1D,
    Convolution2D,
    Correlation2D,
    Convolution3D,
    Correlation3D,
)

operators = {
    "fft": FFT,
    "ifft": IFFT,
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
    "convolve1d": Convolution1D,
    "correlate1d": Correlation1D,
    "convolve2d": Convolution2D,
    "correlate2d": Correlation2D,
    "convolve3d": Convolution3D,
    "correlate3d": Correlation3D,
    "glcm-base": GLCMBase,
    "glcm-asm": GLCMASM,
    "glcm-contrast": GLCMContrast,
    "glcm-correlation": GLCMCorrelation,
    "glcm-variance": GLCMVariance,
    "glcm-energy": GLCMEnergy,
    "glcm-entropy": GLCMEntropy,
    "glcm-mean": GLCMMean,
    "glcm-std": GLCMStandardDeviation,
    "glcm-dissimilarity": GLCMDissimilarity,
    "glcm-homogeneity": GLCMHomogeneity,
}


@auto_scheduler.register_workload
def search_operator(x, y, z, dtype, operator):
    if "glcm" in operator and x*y*z >= 2**17: # volumes larger than that result in indexing problems with GLCM
        x = np.int64(x)
        y = np.int64(y)
        z = np.int64(z)
    X = te.placeholder((x, y, z), name="X", dtype=dtype)
    if operator == "ifft":
        Y = te.placeholder((x, y, z), name="Y", dtype=dtype)
        op = operators[operator](
            computation_context={"x": x, "y": y, "z": z, "X": X, "Y": Y}
        )
    else:
        op = operators[operator](
            computation_context={
                "x": x,
                "y": y,
                "z": z,
                "X": X,
            }
        )

    context = op.computation_context
    tensors = []
    tensors.extend(context["input"])
    tensors.extend(context["result"])
    return tensors


def build_module(args):
    build_profile = {
        c.split("-")[0]: int(c.split("-")[1]) for c in args.config.split("_")
    }
    with open(args.profiles, "r") as f:
        profiles = json.load(f)

    tgt = tvm.target.Target(
        target=profiles[args.arch][build_profile[args.arch]], host="llvm"
    )
    start = perf_counter()
    name = f"{args.operator}_{args.arch}"
    x, y, z = args.x, args.y, args.z
    if "correlate" in args.operator or "convolve" in args.operator:
        if args.operator[-2:] == "1d":
            args.x = 1
            args.y = 1
        elif args.operator[-2:] == "2d":
            args.x = 1
    if build_profile["sch"] == 0:  # Default
        computation_context = {
            "x": args.x if args.x else te.var("x"),
            "y": args.y if args.y else te.var("y"),
            "z": args.z if args.z else te.var("z"),
        }
        computation_context["X"] = te.placeholder(
            (
                computation_context["x"],
                computation_context["y"],
                computation_context["z"],
            ),
            dtype=args.dtype,
            name="X",
        )
        if args.operator in ["ifft"]:
            computation_context["Y"] = te.placeholder(
                (
                    computation_context["x"],
                    computation_context["y"],
                    computation_context["z"],
                ),
                dtype=args.dtype,
                name="Y",
            )

        operator = operators[args.operator](computation_context=computation_context)
        schedule = te.create_schedule(
            [res.op for res in operator.computation_context["result"]]
        )
        build = tvm.build(
            schedule,
            [
                *operator.computation_context["input"],
                *operator.computation_context["result"],
            ],
            tgt,
            name=name,
        )
    elif build_profile["sch"] == 1:  # Ansor
        task = tvm.auto_scheduler.SearchTask(
            func=search_operator,
            args=(args.x, args.y, args.z, args.dtype, args.operator),
            target=tgt,
        )
        ansor_config = profiles["ansor"][build_profile.get("ansor", 0)]

        log_file = os.path.join(args.build_path, "logs", f"{name}_log.json")
        if args.arch == "cpu":
            tune_option = auto_scheduler.TuningOptions(
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                verbose=1,
                **ansor_config["tune_options"],
            )
        elif args.arch == "gpu":
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(
                repeat=1, min_repeat_ms=500, timeout=10
            )
            tune_option = auto_scheduler.TuningOptions(
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                verbose=1,
                runner=measure_ctx.runner,
                **ansor_config["tune_options"],
            )
        s_p = auto_scheduler.SketchPolicy(task, params=ansor_config["sketch_policy"])
        task.tune(tune_option, search_policy=s_p)
        schedule, args_tvm = task.apply_best(log_file)
        build = tvm.build(schedule, args_tvm, tgt)
    else:
        raise NotImplementedError("Scheduler Option not implemented")
    build.export_library(os.path.join(args.build_path, f"{name}.so"))
    print(f"BUILD TOOK: {perf_counter() - start}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--operator",
        help="operator to build",
        type=str,
        choices=operators.keys(),
        default="hilbert",
    )

    parser.add_argument(
        "-a",
        "--arch",
        help="which arch to build",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Config string to use when building",
        type=str,
    )

    parser.add_argument(
        "-x",
        "--x",
        help="x dimension",
        type=int,
        default=None,
    )

    parser.add_argument(
        "-y",
        "--y",
        help="y dimension",
        type=int,
        default=None,
    )

    parser.add_argument(
        "-z",
        "--z",
        help="z dimension",
        type=int,
        default=None,
    )

    parser.add_argument(
        "-d",
        "--dtype",
        help="data type",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )

    parser.add_argument(
        "-p",
        "--profiles",
        help="Build profiles JSON file path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-b", "--build-path", help="path to build folder", type=Path, required=True
    )

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    args = parser.parse_args()
    build_module(args)
