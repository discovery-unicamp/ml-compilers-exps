import jax.numpy as jnp
import jax.scipy as jsci


def first_derivative(X):
    weights = jnp.array([[[-0.5, 0, 0.5]]], dtype=X.dtype)
    X = jnp.pad(X, pad_width=((0, 0), (0, 0), (1, 1)), mode="symmetric")
    X = jsci.signal.correlate(X, weights, mode="valid", method="direct")
    weights = jnp.array([[[0.178947]], [[0.642105]], [[0.178947]]], dtype=X.dtype)
    X = jnp.pad(X, pad_width=((1, 1), (0, 0), (0, 0)), mode="symmetric")
    X = jsci.signal.correlate(X, weights, mode="valid", method="direct")
    weights = jnp.array([[[0.178947], [0.642105], [0.178947]]], dtype=X.dtype)
    X = jnp.pad(X, pad_width=((0, 0), (1, 1), (0, 0)), mode="symmetric")
    X = jsci.signal.correlate(X, weights, mode="valid", method="direct")
    return X
