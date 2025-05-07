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
    
    
