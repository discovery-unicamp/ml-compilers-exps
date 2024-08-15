import argparse
import json
import os
from pathlib import Path

import numpy as np
from dasf_seismic.datasets import DatasetSEGY
from dasf.datasets import DatasetArray, DatasetZarr


datasets = {"SEG-Y": DatasetSEGY, "Zarr": DatasetZarr, "NPY": DatasetArray}


def creator(args):
    with open(args.filename, "r") as f:
        json_dict = json.load(f)
    assert args.name in json_dict

    dataset_configs = json_dict[args.name]
    dataset_loader = datasets[dataset_configs["format"]]
    dataset = (
        dataset_loader("dataset", download=False, root=args.dataset)
        ._lazy_load_cpu()
        ._data
    )

    for sli, sli_conf in dataset_configs["slices"].items():
        sli_size = list(map(int, sli.split("-")))
        for conf in sli_conf:
            if not os.path.exists(os.path.join(args.name, sli, conf["name"])):
                if not os.path.exists(os.path.join(args.name, sli)):
                    os.mkdir(os.path.join(args.name, sli))
                np.save(
                    os.path.join(args.name, sli, conf["name"]),
                    dataset[
                        conf["start"][0] : conf["start"][0] + sli_size[0],
                        conf["start"][1] : conf["start"][1] + sli_size[1],
                        conf["start"][2] : conf["start"][2] + sli_size[2],
                    ],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--filename",
        help="JSON file with slice set",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="dataset path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-n",
        "--name",
        help="dataset name",
        type=str,
        default="parihaka",
    )

    args = parser.parse_args()
    creator(args)
