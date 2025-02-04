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
    run_index = os.path.join(
        os.getcwd(), "experiments", "results_1", exp_id, "index.json"
    )
    run_specs = {
        "CPU": args.cpu and "CPU" in scope,
        "GPU": args.gpu and "GPU" in scope,
        "Baseline": args.baseline,
        "TVM": args.tvm,
        "JAX": args.jax,
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
            os.getcwd(), "experiments", "modules_1", exp_id, f"Build{args.tvm:02d}"
        )

    if run_specs["CPU"]:
        for spec in ["Baseline", "TVM", "JAX"]:
            if run_specs[spec] is not None:
                process = subprocess.run(
                    [
                        f"./scripts/run_{spec.lower()}_cpu_{scope['CPU']['arch']}.sh"
                        if spec == "Baseline"
                        else f"./scripts/run_{spec.lower()}_cpu.sh",
                        csv_file,
                        str(run_specs["repeat"]),
                        os.path.join(os.getcwd(), "data", run_specs["dataset"]),
                        scope["dtype"],
                        str(run_specs["max_threads"]),
                        base_path,
                    ],
                    capture_output=True,
                )
                with open(f"{spec.lower()}_cpu_stdout.log", "w") as f:
                    f.write(process.stdout.decode("ascii"))
                with open(f"{spec.lower()}_cpu_stderr.log", "w") as f:
                    f.write(process.stderr.decode("ascii"))

    if run_specs["GPU"]:
        for spec in ["Baseline", "TVM", "JAX"]:
            if run_specs[spec]:
                process = subprocess.run(
                    [
                        f"./scripts/run_{spec.lower()}_gpu.sh",
                        csv_file,
                        str(run_specs["repeat"]),
                        os.path.join(os.getcwd(), "data", run_specs["dataset"]),
                        scope["dtype"],
                        base_path,
                    ],
                    capture_output=True,
                )
                with open(f"{spec.lower()}_gpu_stdout.log", "w") as f:
                    f.write(process.stdout.decode("ascii"))
                with open(f"{spec.lower()}_gpu_stderr.log", "w") as f:
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
    parser.add_argument(
        "-t", "--tvm", help="Run TVM operators, pass build-id", type=int, default=None
    )
    parser.add_argument("-j", "--jax", help="Run JAX operators", action="store_true")

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
