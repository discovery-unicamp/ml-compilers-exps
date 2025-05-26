import numpy as np

def get_chunks(
    data_shape,
    chunksize,
    overlap,
):
    out_indices = []
    # Calculate the maximum index on each dimension
    max_indices = tuple(
        int(np.ceil(data_shape[i] / chunksize[i])) for i in range(len(data_shape))
    )

    # Generate indices for left upper corner of patches
    for index in np.ndindex(*max_indices):
        corner_index = tuple(index[i] * chunksize[i] for i in range(len(index)))
        out_indices.append(corner_index)

    in_indices = []
    padding_indx = []

    for ind in out_indices:
        curr_i = []
        curr_p = []
        for i, o, c, m in zip(ind, overlap, chunksize, data_shape):
            p = [0, 0]
            c_i = i - o[0]
            if c_i < 0:
                p[0] = abs(c_i)
                c_i = 0
            if i + c + o[1] >= m:
                p[1] = i + c + o[1] - m
            curr_i.append(c_i)
            curr_p.append(tuple(p))
        in_indices.append(tuple(curr_i))
        padding_indx.append(tuple(curr_p))
    return in_indices, out_indices, padding_indx


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

def get_conv_chunksize_overlap(
        rt,
        exp_shape,
):
    chunksize = [
        exp_shape[0] - 2,
        exp_shape[1] - 2,
        exp_shape[2] - 2,
    ]



    overlap = [
        (1, 1),
        (1, 1),
        (1, 1),
    ]
    
    return chunksize, overlap

def get_complex_trace_chunksize_overlap(
        rt,
        exp_shape,
):
    chunksize = [
        exp_shape[0],
        exp_shape[1],
        exp_shape[2] // 2,
    ]



    overlap = [
        (0, 0),
        (0, 0),
        (exp_shape[2] // 4, exp_shape[2] // 4),
    ]

    
    return chunksize, overlap
    
    
