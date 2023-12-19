import argparse
import json
from pathlib import Path
from random import randint


def picker(args):
    with open(args.filename, "r") as f:
        json_dict = json.load(f)
    for dataset_info in json_dict.values():
        shape = dataset_info["shape"]
        slices = dataset_info["slices"].keys()
        for sli in slices:
            samples = []
            sli_size = list(map(int, sli.split("-")))
            assert len(shape) == len(sli_size)
            ranges = [s1 - s2 for s1, s2 in zip(shape, sli_size)]
            for r in ranges:
                assert r >= 0
            for n in range(args.number):
                sample = {
                    "name": f"{n+1}.npy",
                    "start": [randint(0, r) for r in ranges],
                }
                samples.append(sample)

            dataset_info["slices"][sli] = samples
    with open(args.filename, "w") as f:
        json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--filename",
        help="JSON file with preset configurations",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-n",
        "--number",
        help="number of samples",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    picker(args)
