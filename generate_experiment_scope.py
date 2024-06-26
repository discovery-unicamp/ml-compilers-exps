import argparse
import json
import os
import platform
import psutil
import pytz
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cpuinfo


def get_git_revision_hash():
    subprocess.check_output(
        ["git", "config", "--global", "--add", "safe.directory", os.getcwd()]
    )  # bypass git repo ownership that leads to a problem when running inside a Docker container
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def generate_experiment_scope(args):
    exp_dict = {}
    exp_id = str(uuid4())
    if args.cpu:
        info = cpuinfo.get_cpu_info()
        exp_dict["CPU"] = {
            "name": info.get("brand_raw"),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "min_freq": psutil.cpu_freq().min,
            "max_freq": psutil.cpu_freq().max,
            "l1_data_cache_size": info.get("l1_data_cache_size"),
            "l1_instruction_cache_size": info.get("l1_instruction_cache_size"),
            "l2_cache_size": info.get("l2_cache_size"),
            "l2_cache_line_size": info.get("l2_cache_line_size"),
            "l2_cache_associativity": info.get("l2_cache_associativity"),
            "l3_cache_size": info.get("l3_cache_size"),
            "arch": platform.processor(),
            "memory": psutil.virtual_memory().total,
            "flags": info.get("flags"),
        }
    if args.gpu:
        from GPUtil import getGPUs

        gpus = getGPUs()
        exp_dict["GPU"] = [
            {
                "name": gpu.name,
                "driver": gpu.driver,
                "memory": gpu.memoryTotal,
                "id": gpu.id,
            }
            for gpu in gpus
        ]
    exp_dict["data_size"] = args.size
    exp_dict["dtype"] = args.dtype
    exp_dict["commit_hash"] = get_git_revision_hash()
    exp_dict["creation_time"] = (
        datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%d/%m/%Y %H:%M:%S-%Z")
    )
    exp_dict["tag"] = exp_dict["commit_hash"] if args.tag is None else args.tag

    file = (
        os.path.join("experiments", "experiment_index.json")
        if args.file is None
        else args.file
    )
    if os.path.isfile(file):
        with open(file, "r") as f:
            current = json.load(f)
        while exp_id in current:
            exp_id = str(uuid4())
        current[exp_id] = exp_dict
        with open(file, "w") as f:
            json.dump(current, f, indent=4)
    else:
        with open(file, "w") as f:
            json.dump({exp_id: exp_dict}, f, indent=4)

    os.makedirs(os.path.join("experiments", "modules", exp_id))
    os.makedirs(os.path.join("experiments", "results", exp_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", help="JSON file path", type=Path, default=None)

    parser.add_argument("-c", "--cpu", help="CPU will be used", action="store_true")

    parser.add_argument("-g", "--gpu", help="GPU will be used", action="store_true")

    parser.add_argument(
        "-s",
        "--size",
        help="Volume Size",
        type=str,
        choices=["64-64-256", "64-64-512", "128-128-512", "128-128-1024"],
        default="64-64-256",
    )

    parser.add_argument(
        "-d",
        "--dtype",
        help="data type to use",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )

    parser.add_argument(
        "-t",
        "--tag",
        help="add tag to experiment",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    generate_experiment_scope(args)
