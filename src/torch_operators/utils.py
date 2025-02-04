import torch
import torch.nn.functional as F


def first_derivative(X):
    X = X.unsqueeze(0).unsqueeze(0)

    weights = torch.tensor([[[-0.5, 0, 0.5]]], dtype=X.dtype, device=X.device).unsqueeze(0).unsqueeze(0)
    X = F.pad(X, (1, 1, 0, 0, 0, 0), mode="replicate")
    X = F.conv3d(X, weights, padding=0)

    weights = (
        torch.tensor([[[0.178947]], [[0.642105]], [[0.178947]]], dtype=X.dtype, device=X.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    X = F.pad(X, (0, 0, 0, 0, 1, 1), mode="replicate")
    X = F.conv3d(X, weights, padding=0)

    weights = (
        torch.tensor([[[0.178947], [0.642105], [0.178947]]], dtype=X.dtype, device=X.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    X = F.pad(X, (0, 0, 1, 1, 0, 0), mode="replicate")
    X = F.conv3d(X, weights, padding=0)

    return X.squeeze()
