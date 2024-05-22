import jax.numpy as jnp


def L2(y, y_pred, axis=None):
    # equivalent to np.sqrt(np.sum((y-y_pred)**2, axis = axis))
    # frobenius norm is default
    return jnp.linalg.norm(y - y_pred, axis=axis)


def rel_L2(y, y_pred, axis=None):
    # Calculate the relative L2 error.
    return L2(y, y_pred, axis=axis) / jnp.linalg.norm(y, axis=axis)


def mse(y, y_pred, axis=None):
    # Calculate the mean squared error.
    return jnp.mean((y - y_pred) ** 2, axis=axis)


def rmse(y, y_pred, axis=None):
    # Calculate the root mean squared error.
    return jnp.sqrt(mse(y, y_pred, axis))


def nrmse(y, y_pred, axis=None, normalize_by="rms"):
    # Calculate the normalized root mean squared error.
    if normalize_by == "rms":
        norm = jnp.sqrt(jnp.mean(y) ** 2)
    elif normalize_by == "maxmin":
        norm = jnp.max(y, axis=axis) - jnp.min(y, axis=axis)
    elif normalize_by == "std":
        norm = jnp.std(y) ** 2

    return rmse(y, y_pred, axis) / norm
