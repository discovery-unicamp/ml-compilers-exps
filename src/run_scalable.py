import argparse
import csv
import os
from time import perf_counter
from pathlib import Path
from torch_operators.operator_generic import TorchOperator
import zarr
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster



from scalable_integration.utils import get_glcm_chunksize_overlap
from scalable_integration.custom_worker import DaskOperatorWorker


def run_exp(attr, rt, device, input_path, output_path, base_path, dtype, shape):

    shape = tuple(map(int, shape.split("-")))
    print(shape)
    cluster_kwargs = {}
    cluster_kwargs["worker_class"] = DaskOperatorWorker
    cluster_kwargs["rt"] = rt
    cluster_kwargs["attribute"] = attr
    cluster_kwargs["device"] = device
    cluster_kwargs["base_path"] = base_path
    if rt == "torch_c": # torch.compile operators support only one thread per worker and also need to use a multiprocessing module that allows creating child processes from daemons.
        cluster_kwargs["threads_per_worker"] = 1
        os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "aux_repo")
    if rt == "tvm" and not(os.path.exists(os.path.join(base_path, f"{attr}_{device}.so"))):
        raise EnvironmentError("TVM module does not exist")
    if device == "cpu":
        cluster_kwargs["n_workers"] = 1
        cluster = LocalCluster(**cluster_kwargs)
    else:
        cluster = LocalCUDACluster(**cluster_kwargs)

    client = Client(cluster)
    start = perf_counter()

    if "glcm" in attr:
        chunksize, overlap = get_glcm_chunksize_overlap(rt=rt, exp_shape=shape)


    input_data = zarr.open(input_path)
    output_data = zarr.create(
        store=output_path,
        shape=input_data.shape,
        chunks=chunksize,
        dtype=dtype,
    )

    if "glcm" in attr:
        from scalable_integration.texture import glcm_base
        tasks = glcm_base(
            rt=rt,
            input_data=input_data,
            output_data=output_data,
            chunksize=chunksize,
            overlap=overlap,
            dtype=dtype,
            device=device,
        )

    client.compute(tasks, sync=True)
    end = perf_counter()
    client.shutdown()
    return end - start




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        help="device to use, CPU or GPU",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
    )

    parser.add_argument(
        "-u",
        "--attribute",
        help="attribute to use",
        type=str,
        choices=[
            "envelope",
            "inst-phase",
            "cos-inst-phase",
            "relative-amplitude-change",
            "amplitude-acceleration",
            "inst-bandwidth",
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
    )

    parser.add_argument(
        "-r",
        "--runtime",
        help="runtime to use",
        type=str,
        choices=["tvm", "baseline", "torch_c", "torch_n", "jax"],
        default="baseline",
    )

    parser.add_argument(
        "-b",
        "--base-path",
        help="path to module files",
        type=Path,
        required=True,
    )

    parser.add_argument("-f", "--file", help="CSV file path", type=Path, required=True)


    parser.add_argument(
        "-p",
        "--input-path",
        help="input path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-path",
        help="output path",
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

    parser.add_argument(
        "-s",
        "--shape",
        help="data shape",
        type=str,
        required=True
    )

    parser.add_argument(
        "-i",
        "--id",
        help="TVM build identifier",
        type=int,
        default=1,
    )
    
    args = parser.parse_args()
    exec_time = run_exp(
        attr=args.attribute,
        rt=args.runtime,
        device=args.arch,
        input_path=args.input_path,
        output_path=args.output_path,
        base_path=args.base_path,
        dtype=args.dtype,
        shape=args.shape,
    )

    with open(args.file, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                args.attribute,
                f"{args.runtime}_{args.arch}_{args.id}",
                os.getenv("OMP_NUM_THREADS", 0),
                exec_time
            ]
        )
