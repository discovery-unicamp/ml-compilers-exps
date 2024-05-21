import argparse
import json
import os
from pathlib import Path

import subprocess

schedules = {
    "default": 0,
    "ansor": 1,
    "autotvm": 2,
}

attr_list = [
    # # "fft",
    # # "ifft",
    # # "hilbert",
    # "envelope",
    # "inst-phase",
    # "cos-inst-phase",
    # "relative-amplitude-change",
    # "amplitude-acceleration",
    # # "inst-frequency",
    # "inst-bandwidth",
    # # "dominant-frequency",
    # # "frequency-change",
    # # "sweetness",
    # # "quality-factor",
    # # "response-phase",
    # # "response-frequency",
    # # "response-amplitude",
    # # "apparent-polarity",
    "convolve1d",
    "correlate1d",
    "convolve2d",
    "correlate2d",
    "convolve3d",
    "correlate3d",
    # "glcm-base",
#     "glcm-asm",
#     "glcm-contrast",
#     "glcm-correlation",
#     "glcm-variance",
#     "glcm-energy",
#     "glcm-entropy",
#     "glcm-mean",
#     "glcm-std",
#     "glcm-dissimilarity",
#     "glcm-homogeneity"
]


def build_modules(args):
    with open(args.file, "r") as f:
        scope = json.load(f)
        exp_id = (
            args.experiment_id
            if args.experiment_id is not None
            else list(scope.keys())[-1]
        )
        scope = scope[exp_id]
    build_index = os.path.join(
        os.getcwd(), "experiments", "modules", exp_id, "index.json"
    )

    build_specs = ""
    arch_list = []
    if args.cpu != -1 and "CPU" in scope:
        build_specs += f"cpu-{args.cpu}_"
        arch_list.append("cpu")
    if (
        args.gpu != -1 and "GPU" in scope and args.scheduler != "default"
    ):  # CUDA has no default schedule
        build_specs += f"gpu-{args.gpu}_"
        arch_list.append("gpu")
    if args.scheduler == "ansor":
        build_specs += f"sch-{schedules[args.scheduler]}_ansor-{args.ansor}"
    else:
        build_specs += f"sch-{schedules[args.scheduler]}"

    if os.path.isfile(build_index):
        with open(build_index, "r") as f:
            content = json.load(f)
            curr_run = len(content.keys()) + 1
        with open(build_index, "w") as f:
            content[curr_run] = build_specs
            json.dump(content, f, indent=4)
    else:
        curr_run = 1
        with open(build_index, "w") as f:
            json.dump({curr_run: build_specs}, f, indent=4)
    build_folder = os.path.join(
        os.getcwd(), "experiments", "modules", exp_id, f"Build{curr_run:02d}"
    )
    os.mkdir(build_folder)
    os.mkdir(os.path.join(build_folder, "logs"))
    x, y, z = scope["data_size"].split("-")

    for attr in attr_list:
        for arch in arch_list:
            process = subprocess.run(
                [
                    "python",
                    os.path.join("src", "build_tvm_module.py"),
                    "--operator",
                    attr,
                    "--arch",
                    arch,
                    "--config",
                    build_specs,
                    "--x",
                    x,
                    "--y",
                    y,
                    "--z",
                    z,
                    "--dtype",
                    scope["dtype"],
                    "--profiles",
                    args.profiles,
                    "--build-path",
                    build_folder,
                ],
                capture_output=True,
            )
            with open(
                os.path.join(build_folder, "logs", f"{attr}_{arch}_stdout.log"), "w"
            ) as f:
                f.write(process.stdout.decode("ascii"))
            with open(
                os.path.join(build_folder, "logs", f"{attr}_{arch}_stderr.log"), "w"
            ) as f:
                f.write(process.stderr.decode("ascii"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        help="Experiment Index JSON file path",
        type=Path,
        default=os.path.join("experiments", "experiment_index.json"),
    )

    parser.add_argument(
        "-p",
        "--profiles",
        help="Build profiles JSON file path",
        type=Path,
        default=os.path.join(os.getcwd(), "experiments", "build_profiles.json"),
    )

    parser.add_argument(
        "-c",
        "--cpu",
        help="CPU build profile, if set to -1, will be ignored. Check experiments/build_profiles.json",
        default=-1,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help="GPU build profile, if set to -1, will be ignored. Check experiments/build_profiles.json",
        default=-1,
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        help="scheduler to use",
        choices=["default", "ansor", "autotvm"],
        default="default",
    )
    parser.add_argument(
        "-a",
        "--ansor",
        help="Ansor schedule search policy, relevant only if Ansor is the selected scheduler. Check experiments/build_profiles.json",
        default=0,
    )

    parser.add_argument(
        "-e",
        "--experiment-id",
        help="Experiment ID to use, if not set defaults to last on the file",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    build_modules(args)
