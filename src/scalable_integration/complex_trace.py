import dask
import numpy as np
import cupy as cp
import tvm
import jax
import torch
from scalable_integration.utils import get_chunks
from scalable_integration.custom_worker import get_operator
jax.config.update("jax_enable_x64", True)


def complex_trace_base(
    rt,
    input_data,
    output_data,
    chunksize,
    overlap,
    dtype,
    device,
):

    in_ind, out_ind, padding = get_chunks(
        data_shape=input_data.shape,
        chunksize=chunksize,
        overlap=overlap
    )

    task_funcs = {
        "tvm": complex_trace_tvm,
        "baseline": complex_trace_baseline,
        "torch_c": complex_trace_torch_c,
        "torch_n": complex_trace_torch_n,
        "jax": complex_trace_jax,
    }

    task_func = task_funcs[rt]

    tasks = [
        task_func(
            input_data=input_data,
            output_data=output_data,
            indx=i,
            out_indx=out_i,
            chunksize=chunksize,
            pad_width=p,
            overlap=overlap,
            dtype=dtype,
            device=device,
        )
        for i, out_i, p in zip(in_ind, out_ind, padding)
    ]

    return tasks, chunksize

    

@dask.delayed
def complex_trace_tvm(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=0
    )

    operator = get_operator()

    data_tvm = tvm.nd.array(chunk, device=operator._dev)
    res = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=operator._dev)
    operator.transform(data_tvm, res)

    res = res.numpy()


    useful_slice = [
        min(c, o - i)
        for i, c, o in zip(out_indx, chunksize, output_data.shape)
    ]
    out_sli = tuple(
        slice(i, i + u)
        for i, u in zip(out_indx, useful_slice)
    )

    res_sli = tuple(
        slice(o[0], o[0] + u)
        for u, o in zip(useful_slice, overlap)
    )

    output_data[out_sli] = res[res_sli]

@dask.delayed
def complex_trace_baseline(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=0
    )
    operator = get_operator()

    if device == "cpu":
        res = operator._transform_cpu(chunk)
    else:
        chunk = cp.asarray(chunk)
        res = operator._transform_gpu(chunk).get()


    useful_slice = [
        min(c, o - i)
        for i, c, o in zip(out_indx, chunksize, output_data.shape)
    ]
    out_sli = tuple(
        slice(i, i + u)
        for i, u in zip(out_indx, useful_slice)
    )

    res_sli = tuple(
        slice(o[0], o[0] + u)
        for u, o in zip(useful_slice, overlap)
    )

    output_data[out_sli] = res[res_sli]

@dask.delayed
def complex_trace_jax(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=0
    )
    chunk = jax.device_put(chunk, device=jax.devices(device)[0])

    operator = get_operator()
    if device == "cpu":
        res = operator._transform_cpu(chunk)
    else:
        res = operator._transform_gpu(chunk)
    res = np.asarray(res)


    useful_slice = [
        min(c, o - i)
        for i, c, o in zip(out_indx, chunksize, output_data.shape)
    ]
    out_sli = tuple(
        slice(i, i + u)
        for i, u in zip(out_indx, useful_slice)
    )

    res_sli = tuple(
        slice(o[0], o[0] + u)
        for u, o in zip(useful_slice, overlap)
    )

    output_data[out_sli] = res[res_sli]

@dask.delayed
def complex_trace_torch_c(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=0
    )
    chunk = torch.from_numpy(chunk).to(torch.device("cpu" if device == "cpu" else "cuda"))
    operator = get_operator()
    if device == "cpu":
        res = operator._transform_cpu(chunk)
    else:
        res = operator._transform_gpu(chunk).cpu()
    res = res.numpy()


    useful_slice = [
        min(c, o - i)
        for i, c, o in zip(out_indx, chunksize, output_data.shape)
    ]
    out_sli = tuple(
        slice(i, i + u)
        for i, u in zip(out_indx, useful_slice)
    )

    res_sli = tuple(
        slice(o[0], o[0] + u)
        for u, o in zip(useful_slice, overlap)
    )

    output_data[out_sli] = res[res_sli]


@dask.delayed
def complex_trace_torch_n(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=0
    )
    chunk = torch.from_numpy(chunk).to(torch.device("cpu" if device == "cpu" else "cuda"))
    operator = get_operator()
    if device == "cpu":
        res = operator._nocompile_cpu(chunk)
    else:
        res = operator._nocompile_gpu(chunk).cpu()
    res = res.numpy()


    useful_slice = [
        min(c, o - i)
        for i, c, o in zip(out_indx, chunksize, output_data.shape)
    ]
    out_sli = tuple(
        slice(i, i + u)
        for i, u in zip(out_indx, useful_slice)
    )

    res_sli = tuple(
        slice(o[0], o[0] + u)
        for u, o in zip(useful_slice, overlap)
    )

    output_data[out_sli] = res[res_sli]