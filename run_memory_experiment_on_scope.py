import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import subprocess

attr_classes = {
    "glcm": [
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
    ],
    "complex_trace": [
        "fft",
        "envelope",
        "inst-phase",
        "cos-inst-phase",
        "relative-amplitude-change",
        "amplitude-acceleration",
        "inst-bandwidth",
    ]
}

shape_map = {
    "64-64-256": "complex_trace",
    "64-64-512": "complex_trace",
    "128-128-512": "complex_trace",
    "128-128-1024": "complex_trace",
    "32-32-32": "glcm",
    "64-64-64": "glcm",
    "128-128-128": "glcm",
}

def run_experiments(args):
    with open(args.file, "r") as f:
        scope = json.load(f)
        exp_id = (
            args.experiment_id
            if args.experiment_id is not None
            else list(scope.keys())[-1]
        )
        scope = scope[exp_id]
    run_dir = os.path.join(
        os.getcwd(), "experiments", "results_1", exp_id
    )
    run_specs = {
        "CPU": args.cpu,
        "GPU": args.gpu,
        "Baseline": args.baseline,
        "TVM": args.tvm,
        "JAX": args.jax,
        "TORCH_C": args.torch_compile,
        "TORCH_N": args.torch_nocompile,
        "max_threads": args.max_threads,
    }

    
    csv_file = os.path.join(
        os.getcwd(), "experiments", "results_1", exp_id, f"Memory_{args.strategy}.csv"
    )
    gpu_csv_file = os.path.join(
        os.getcwd(), "experiments", "results_1", exp_id, f"Memory_gpu.csv"
    )
    if run_specs["CPU"]:
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "attr",
                    "dataset_id",
                    "env",
                    "threads",
                    "mem_1",
                    "mem_2",
                ]
            )

    if run_specs["GPU"]:
        with open(gpu_csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "attr",
                    "dataset_id",
                    "env",
                    "threads",
                    "pageable",
                    "pinned",
                    "device",
                    "array",
                    "managed",
                    "device_static",
                    "managed_static",
                    "unknown"
                ]
            )
    base_path = ""
    if args.tvm != None:
        base_path = os.path.join(
            os.getcwd(), "experiments", "modules_1", exp_id
        )

    for dataset_id in range(1, 6):
        dataset_id = str(dataset_id)
        dataset_path = os.path.join(os.getcwd(), "data", "parihaka", scope["data_size"], f"{dataset_id}.npy")
        for attr in attr_classes[shape_map[scope["data_size"]]]:
            if run_specs["CPU"]:
                for spec in ["Baseline", "TVM", "JAX", "TORCH_C", "TORCH_N"]:
                    if run_specs[spec]:
                        process = subprocess.run(
                            [
                                f"./scripts/memory_footprint_cpu.sh",
                                csv_file,
                                attr,
                                f"{spec.lower()}_cpu_1" if spec == "TVM" else f"{spec.lower()}_cpu",
                                scope["dtype"],
                                str(run_specs["max_threads"]),
                                base_path,
                                dataset_id,
                                dataset_path,
                                args.strategy,
                            ],
                            capture_output=True,
                        )
                        with open(f"mem_log/{args.strategy}_mem_{spec.lower()}_{attr}_cpu_stdout_{exp_id}_{dataset_id}.log", "w") as f:
                            f.write(process.stdout.decode("ascii"))
                        with open(f"mem_log/{args.strategy}_mem_{spec.lower()}_{attr}_cpu_stderr_{exp_id}_{dataset_id}.log", "w") as f:
                            f.write(process.stderr.decode("ascii"))
                        
                        if spec == "TVM":
                            process = subprocess.run(
                                [
                                    f"./scripts/memory_footprint_cpu.sh",
                                    csv_file,
                                    attr,
                                    f"{spec.lower()}_cpu_2",
                                    scope["dtype"],
                                    str(run_specs["max_threads"]),
                                    base_path,
                                    dataset_id,
                                    dataset_path,
                                    args.strategy,
                                ],
                                capture_output=True,
                            )
                            with open(f"mem_log/{args.strategy}_mem_{spec.lower()}_{attr}_cpu_2_stdout_{exp_id}_{dataset_id}.log", "w") as f:
                                f.write(process.stdout.decode("ascii"))
                            with open(f"mem_log/{args.strategy}_mem_{spec.lower()}_{attr}_cpu_2_stderr_{exp_id}_{dataset_id}.log", "w") as f:
                                f.write(process.stderr.decode("ascii"))

            if run_specs["GPU"]:
                for spec in ["Baseline", "TVM", "JAX", "TORCH_C", "TORCH_N"]:
                    if run_specs[spec]:
                        process = subprocess.run(
                            [
                                f"./scripts/memory_footprint_gpu.sh",
                                gpu_csv_file,
                                attr,
                                f"{spec.lower()}_gpu",
                                scope["dtype"],
                                str(run_specs["max_threads"]),
                                base_path,
                                dataset_id,
                                dataset_path,
                            ],
                            capture_output=True,
                        )
                        with open(f"mem_log/gpu_mem_{spec.lower()}_{attr}_gpu_stdout_{exp_id}_{dataset_id}.log", "w") as f:
                            f.write(process.stdout.decode("ascii"))
                        with open(f"mem_log/gpu_mem_{spec.lower()}_{attr}_gpu_stderr_{exp_id}_{dataset_id}.log", "w") as f:
                            f.write(process.stderr.decode("ascii"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        help="Experiment Index JSON file path",
        type=Path,
        default=os.path.join("experiments", "index.json"),
    )

    parser.add_argument(
        "-c",
        "--cpu",
        help="CPU will be used, needs to be set on scope as well",
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help="GPU will be used, needs to be set on scope as well",
        action="store_true",
    )
    parser.add_argument("-b", "--baseline", help="Run baselines", action="store_true")
    parser.add_argument("-t", "--tvm", help="Run TVM operators",  action="store_true")

    parser.add_argument("-j", "--jax", help="Run JAX operators",  action="store_true")

    parser.add_argument("-o", "--torch-compile", help="Run Torch Compile operators", action="store_true")

    parser.add_argument("-n", "--torch-nocompile", help="Run Torch operators without compile", action="store_true")

    parser.add_argument(
        "-m",
        "--max-threads",
        help="Max number of threads (OMP_NUM_THREADS) to test, will set OMP_NUM_THREADS",
        type=int,
        default=1,
    )

    parser.add_argument("-s", "--strategy", help="which profiling startegy to use", default="memray", choices=["memray", "time", "mprof"])

    parser.add_argument(
        "-e",
        "--experiment-id",
        help="Experiment ID to use, if not set defaults to last on the file",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    run_experiments(args)
