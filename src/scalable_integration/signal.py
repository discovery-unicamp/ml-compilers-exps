import dask
import numpy as np
import cupy as cp
import tvm
import jax
import torch
from scalable_integration.utils import get_chunks
from scalable_integration.custom_worker import get_operator
jax.config.update("jax_enable_x64", True)

weights = np.array(
    [
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, 26, 1], [1, 1, 1]],
        [[-1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ]
)

weights_tvm = np.array(
    [
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, 26, 1], [1, 1, 1]],
        [[-1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ]
)



def conv_base(
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
        "tvm": conv_tvm,
        "baseline": conv_baseline,
        "torch_c": conv_torch_c,
        "torch_n": conv_torch_n,
        "jax": conv_jax,
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
def conv_tvm(
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
    weight_tvm = tvm.nd.array(weights_tvm.astype(dtype), device=operator._dev)
    res = tvm.nd.empty(data_tvm.shape, dtype=data_tvm.dtype, device=operator._dev)
    operator.transform(data_tvm, weight_tvm, res)

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
def conv_baseline(
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
    weight = weights.astype(dtype)
    operator = get_operator()

    if device == "cpu":
        res = operator._transform_cpu(chunk, weight)
    else:
        chunk = cp.asarray(chunk)
        weight =cp.asarray(weight)
        res = operator._transform_gpu(chunk, weight).get()


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
def conv_jax(
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
    weight = jax.device_put(weights.astype(dtype), device=jax.devices(device)[0])

    operator = get_operator()
    if device == "cpu":
        res = operator._transform_cpu(chunk, weight)
    else:
        res = operator._transform_gpu(chunk, weight)
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
def conv_torch_c(
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
    weight = torch.from_numpy(weights.astype(dtype)).to(torch.device("cpu" if device == "cpu" else "cuda"))
    operator = get_operator()
    if device == "cpu":
        res = operator._transform_cpu(chunk, weight)
    else:
        res = operator._transform_gpu(chunk, weight).cpu()
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
def conv_torch_n(
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
    weight = torch.from_numpy(weights.astype(dtype)).to(torch.device("cpu" if device == "cpu" else "cuda"))
    operator = get_operator()
    if device == "cpu":
        res = operator._nocompile_cpu(chunk, weight)
    else:
        res = operator._nocompile_gpu(chunk, weight).cpu()
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
