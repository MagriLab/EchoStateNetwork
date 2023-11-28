# Input weights generation methods
import jax
import jax.numpy as jnp
# from jax.experimental import sparse

def sparse_random(W_in_shape: tuple, W_in_seeds: list):
    """
    Create the input weights matrix. Inputs are not connected, except for the parameters.

    Args:
        W_in_shape (tuple): (N_reservoir, N_inputs + N_input_bias + N_param_dim)
        W_in_seeds (list): A list of seeds for the random generators; one for the column index, one for uniform sampling

    Returns:
        W_in: Sparse matrix containing the input weights
    """
    
    W_in = jnp.zeros(W_in_shape)
    rnd0, rnd1 = jax.random.split(jax.random.PRNGKey(W_in_seeds[0]), 2)

    # only one element different from zero sample from the uniform distribution
    rnd_idx = jax.random.randint(rnd0, shape=(
        W_in.shape[0],), minval=0, maxval=W_in.shape[1])
    row_idx = jnp.arange(0, W_in.shape[0])
    uniform_values = jax.random.uniform(
        rnd1, shape=(W_in.shape[0],), minval=-1.0, maxval=1.0)

    # Set values in the matrix at the specified indices
    W_in = W_in.at[row_idx, rnd_idx].set(uniform_values)
    # W_in = sparse.csr_fromdense(W_in)
    return W_in


def sparse_grouped(W_in_shape: tuple, W_in_seeds: list):
    """
    Create a grouped input weights matrix. The inputs are not connected, but they are grouped within the matrix.

    Args:
        W_in_shape (tuple): (N_reservoir, N_inputs + N_input_bias + N_param_dim)
        W_in (list): A list of seeds for the random generators; one for the column index, one for uniform sampling

    Returns:
        BCOO: Sparse matrix containing the grouped input weights
    """

    W_in = jnp.zeros(W_in_shape)
    rnd0 = jax.random.split(jax.random.PRNGKey(W_in_seeds[0]), 1)

    # Generate row and column indices
    row_idx = jnp.arange(0, W_in.shape[0])
    column_idx = jnp.floor(
        row_idx * (W_in_shape[1]) / W_in_shape[0]).astype(int)
    uniform_values = jax.random.uniform(
        rnd0, shape=(W_in.shape[0],), minval=-1.0, maxval=1.0)

    # Set values in the matrix at the specified indices
    W_in = W_in.at[row_idx, column_idx].set(uniform_values)
    # W_in = sparse.csr_fromdense(W_in)
    return W_in


def dense(W_in_shape: tuple, W_in_seeds: list):
    """
    Create a dense input weights matrix. All inputs are connected.

    Args:
        W_in_shape (tuple): (N_reservoir, N_inputs + N_input_bias + N_param_dim)
        W_in (list): A list of seeds for the random generators

    Returns:
        jnp.ndarray: Dense matrix containing the input weights
    """

    # Generate random matrix with uniform values
    rnd_key = jax.random.split(jax.random.PRNGKey(W_in_seeds[0]), 1)
    W_in = jax.random.uniform(
        rnd_key, shape=W_in_shape, minval=-1.0, maxval=1.0)

    return W_in
