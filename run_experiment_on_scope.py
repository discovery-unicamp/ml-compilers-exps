import argparse
import csv
import json
import os
from pathlib import Path

import subprocess


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
        run_dir, "index.json"
    )
    run_specs = {
        "CPU": args.cpu,
        "GPU": args.gpu,
        "Baseline": args.baseline,
        "TVM": args.tvm,
        "JAX": args.jax,
        "TORCH_C": args.torch_compile,
        "TORCH_N": args.torch_nocompile,
        "repeat": args.repeat,
        "dataset": os.path.join(args.dataset, scope["data_size"], f"{args.sample}.npy"),
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
        os.getcwd(), "experiments", "results_1", exp_id, f"Run{curr_run:02d}.csv"
    )
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "attr",
                "env",
                "threads",
                *[f"d1t{i+1}" for i in range(args.repeat)],
                *[f"d2t{i+1}" for i in range(args.repeat)],
            ]
        )
    base_path = ""
    if args.tvm != None:
        base_path = os.path.join(
            os.getcwd(), "experiments", "modules_1", exp_id
        )

    if run_specs["CPU"]:
        for spec in ["Baseline", "TVM", "JAX", "TORCH_C", "TORCH_N"]:
            if run_specs[spec]:
                process = subprocess.run(
                    [
                        f"./scripts/run_{spec.lower()}_cpu.sh"
                        if spec == "Baseline"
                        else f"./scripts/run_{spec.lower()}_cpu.sh",
                        csv_file,
                        str(run_specs["repeat"]),
                        os.path.join(os.getcwd(), "data", run_specs["dataset"]),
                        scope["dtype"],
                        str(run_specs["max_threads"]),
                        os.path.join(base_path, "Build01"),
                        "1"
                    ],
                    capture_output=True,
                )
                with open(f"{spec.lower()}_cpu_stdout_{exp_id}.log", "w") as f:
                    f.write(process.stdout.decode("ascii"))
                with open(f"{spec.lower()}_cpu_stderr_{exp_id}.log", "w") as f:
                    f.write(process.stderr.decode("ascii"))
                
                if spec == "TVM":
                    process = subprocess.run(
                        [
                            f"./scripts/run_{spec.lower()}_cpu.sh"
                            if spec == "Baseline"
                            else f"./scripts/run_{spec.lower()}_cpu.sh",
                            csv_file,
                            str(run_specs["repeat"]),
                            os.path.join(os.getcwd(), "data", run_specs["dataset"]),
                            scope["dtype"],
                            str(run_specs["max_threads"]),
                            os.path.join(base_path, "Build02"),
                            "2"
                        ],
                        capture_output=True,
                    )
                    with open(f"{spec.lower()}_2_cpu_stdout_{exp_id}.log", "w") as f:
                        f.write(process.stdout.decode("ascii"))
                    with open(f"{spec.lower()}_2_cpu_stderr_{exp_id}.log", "w") as f:
                        f.write(process.stderr.decode("ascii"))

    if run_specs["GPU"]:
        for spec in ["Baseline", "TVM", "JAX", "TORCH_C", "TORCH_N"]:
            if run_specs[spec]:
                process = subprocess.run(
                    [
                        f"./scripts/run_{spec.lower()}_gpu.sh",
                        csv_file,
                        str(run_specs["repeat"]),
                        os.path.join(os.getcwd(), "data", run_specs["dataset"]),
                        scope["dtype"],
                        os.path.join(base_path, "Build03"),
                        "3"
                    ],
                    capture_output=True,
                )
                with open(f"{spec.lower()}_gpu_stdout_{exp_id}.log", "w") as f:
                    f.write(process.stdout.decode("ascii"))
                with open(f"{spec.lower()}_gpu_stderr_{exp_id}.log", "w") as f:
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
        "-d",
        "--dataset",
        help="dataset to use",
        type=str,
        choices=["parihaka"],
        default="parihaka",
    )

    parser.add_argument(
        "-s",
        "--sample",
        help="dataset sample to use",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
    )

    parser.add_argument(
        "-r",
        "--repeat",
        help="Number of timeit repeat samples",
        type=int,
        default=5,
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
