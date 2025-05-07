import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import subprocess

attr_classes = {
    "glcm": [
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
    run_index = os.path.join(
        run_dir, "index_scalable.json"
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
    if os.path.isfile(run_index):
        with open(run_index, "r") as f:
            content = json.load(f)
            curr_run = len(content.keys()) + 1
        with open(run_index, "w") as f:
            content[curr_run] = run_specs
            json.dump(content, f, indent=4)
    else:
        curr_run = 1
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        with open(run_index, "w") as f:
            json.dump({curr_run: run_specs}, f, indent=4)

    csv_file = os.path.join(
        os.getcwd(), "experiments", "results_1", exp_id, f"RunScalable{curr_run:02d}.csv"
    )
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "attr",
                "env",
                "threads",
                "time",
            ]
        )
    base_path = ""
    if args.tvm != None:
        base_path = os.path.join(
            os.getcwd(), "experiments", "modules_1", exp_id
        )

    for attr in attr_classes[shape_map[scope["data_size"]]]:
        if run_specs["CPU"]:
            for spec in ["TVM", "JAX", "TORCH_C", "TORCH_N"]:
                if run_specs[spec]:
                    output_path = os.path.join(args.output_path, f"{attr}+{spec}+cpu+{exp_id}.zarr")
                    rm_aux_path = Path(output_path)
                    if rm_aux_path.exists():
                        shutil.rmtree(rm_aux_path)
                    process = subprocess.run(
                        [
                            f"./scripts/run_scalable_cpu.sh",
                            csv_file,
                            attr,
                            spec.lower(),
                            scope["data_size"],
                            scope["dtype"],
                            str(run_specs["max_threads"]),
                            os.path.join(base_path, "Build01"),
                            "1",
                            args.input_path,
                            output_path
                        ],
                        capture_output=True,
                    )
                    with open(f"scalable_{spec.lower()}_{attr}_cpu_stdout_{exp_id}.log", "w") as f:
                        f.write(process.stdout.decode("ascii"))
                    with open(f"scalable_{spec.lower()}_{attr}_cpu_stderr_{exp_id}.log", "w") as f:
                        f.write(process.stderr.decode("ascii"))
                    
                    if spec == "TVM":
                        output_path = os.path.join(args.output_path, f"{attr}+{spec}+cpu_2+{exp_id}.zarr")
                        rm_aux_path = Path(output_path)
                        if rm_aux_path.exists():
                            shutil.rmtree(rm_aux_path)
                        process = subprocess.run(
                            [
                                f"./scripts/run_scalable_cpu.sh",
                                csv_file,
                                attr,
                                spec.lower(),
                                scope["data_size"],
                                scope["dtype"],
                                str(run_specs["max_threads"]),
                                os.path.join(base_path, "Build02"),
                                "2",
                                args.input_path,
                                output_path
                            ],
                            capture_output=True,
                        )
                        with open(f"scalable_{spec.lower()}_{attr}_cpu_2_stdout_{exp_id}.log", "w") as f:
                            f.write(process.stdout.decode("ascii"))
                        with open(f"scalable_{spec.lower()}_{attr}_cpu_2_stderr_{exp_id}.log", "w") as f:
                            f.write(process.stderr.decode("ascii"))

        if run_specs["GPU"]:
            for spec in ["Baseline", "TVM", "JAX", "TORCH_C", "TORCH_N"]:
                if run_specs[spec]:
                    output_path = os.path.join(args.output_path, f"{attr}+{spec}+gpu+{exp_id}.zarr")
                    rm_aux_path = Path(output_path)
                    if rm_aux_path.exists():
                        shutil.rmtree(rm_aux_path)
                    process = subprocess.run(
                        [
                            f"./scripts/run_scalable_gpu.sh",
                            csv_file,
                            attr,
                            spec.lower(),
                            scope["data_size"],
                            scope["dtype"],
                            str(run_specs["max_threads"]),
                            os.path.join(base_path, "Build03"),
                            "3",
                            args.input_path,
                            output_path
                        ],
                        capture_output=True,
                    )
                    with open(f"scalable_{spec.lower()}_{attr}_gpu_stdout_{exp_id}.log", "w") as f:
                        f.write(process.stdout.decode("ascii"))
                    with open(f"scalable_{spec.lower()}_{attr}_gpu_stderr_{exp_id}.log", "w") as f:
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
        "-i",
        "--input-path",
        help="Zarr input data path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-u",
        "--output-path",
        help="directory to save outputs",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-m",
        "--max-threads",
        help="Max number of threads (OMP_NUM_THREADS) to test, will set OMP_NUM_THREADS",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-e",
        "--experiment-id",
        help="Experiment ID to use, if not set defaults to last on the file",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    run_experiments(args)
