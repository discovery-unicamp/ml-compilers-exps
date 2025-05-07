from time import perf_counter
import dask
import dask.array as da
import numpy as np
import cupy as cp
import tvm
import jax
import torch
from scalable_integration.utils import get_chunks
from scalable_integration.custom_worker import get_operator
jax.config.update("jax_enable_x64", True)



def get_glcm_chunksize_overlap(
        rt,
        exp_shape,
        window=7,
):
    hw = window//2
    chunksize = [
        exp_shape[0],
        exp_shape[1] - 2*hw,
        exp_shape[2] - 2*hw,
    ]



    overlap = [
        (0, 0),
        (hw, hw),
        (hw, hw),
    ]

    if rt == "tvm":
        chunksize[0] = chunksize[0] - 1
        overlap[0] = (0, 1)
    
    return chunksize, overlap

def glcm_base(
    rt,
    input_data,
    output_data,
    chunksize,
    overlap,
    dtype,
    device,
):
    glb_mi = da.min(input_data)
    glb_ma = da.max(input_data)
    glb_mi, glb_ma = dask.compute(glb_mi, glb_ma)
    in_ind, out_ind, padding = get_chunks(
        data_shape=input_data.shape,
        chunksize=chunksize,
        overlap=overlap
    )

    task_funcs = {
        "tvm": glcm_tvm,
        "baseline": glcm_baseline,
        "torch_c": glcm_torch_c,
        "torch_n": glcm_torch_n,
        "jax": glcm_jax,
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
            glb_mi=glb_mi,
            glb_ma=glb_ma,
            dtype=dtype,
            device=device,
        )
        for i, out_i, p in zip(in_ind, out_ind, padding)
    ]

    return tasks, chunksize

    

@dask.delayed
def glcm_tvm(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    glb_mi,
    glb_ma,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=glb_mi
    )
    if chunk.shape != (32, 32, 32):
        msg = f"CHUNK BAD {chunk.shape} {indx} {out_indx} {overlap} {glb_mi} {glb_ma} {pad_width}"
        raise EnvironmentError(msg)

    chunk[-1, -1, -1] = glb_ma
    chunk[-1, -1, -2] = glb_mi
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
def glcm_baseline(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    glb_mi,
    glb_ma,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=glb_mi
    )
    operator = get_operator()

    if device == "cpu":
        res = operator._transform_cpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma)
    else:
        chunk = cp.asarray(chunk)
        res = operator._transform_gpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma).get()


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
def glcm_jax(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    glb_mi,
    glb_ma,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=glb_mi
    )
    chunk = jax.device_put(chunk, device=jax.devices(device)[0])

    operator = get_operator()
    if device == "cpu":
        res = operator._transform_cpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma)
    else:
        res = operator._transform_gpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma)
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
def glcm_torch_c(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    glb_mi,
    glb_ma,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=glb_mi
    )
    chunk = torch.from_numpy(chunk).to(torch.device("cpu" if device == "cpu" else "cuda"))
    operator = get_operator()
    if device == "cpu":
        chunk = torch.from_numpy(chunk).to(torch.device("cpu"))
        res = operator._transform_cpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma)
    else:
        chunk = torch.from_numpy(chunk).to(torch.device("cuda"))
        res = operator._transform_gpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma)
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
def glcm_torch_n(
    input_data,
    output_data,
    indx,
    out_indx,
    chunksize,
    pad_width,
    overlap,
    glb_mi,
    glb_ma,
    dtype,
    device
):
    sli = tuple(
        slice(i, i + c + o[0] + o[1] - p[0] - p[1])
        for i, c, p, o in zip(indx, chunksize, pad_width, overlap)
    )
    chunk = input_data[sli].astype(dtype)
    chunk = np.pad(
        chunk, pad_width=pad_width, mode="constant", constant_values=glb_mi
    )
    chunk = torch.from_numpy(chunk).to(torch.device("cpu" if device == "cpu" else "cuda"))
    operator = get_operator()
    if device == "cpu":
        chunk = torch.from_numpy(chunk).to(torch.device("cpu"))
        res = operator._nocompile_cpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma)
    else:
        chunk = torch.from_numpy(chunk).to(torch.device("cuda"))
        res = operator._nocompile_gpu(chunk, glb_mi=glb_mi, glb_ma=glb_ma)
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