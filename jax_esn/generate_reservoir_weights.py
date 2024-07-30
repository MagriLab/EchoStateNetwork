# Reservoir weights generation methods
import jax
import jax.numpy as jnp
# from jax.experimental import sparse


def erdos_renyi1(W_shape, sparseness, W_seeds):
    """
    Create the reservoir weights matrix according to Erdos-Renyi network.

    Args:
        W_shape (tuple): Shape of the matrix (rows, columns)
        sparseness (float): Sparseness parameter
        W_seeds (list): A list of seeds for the random generators; one for the connections, one for uniform sampling of weights
    Returns:
        W: sparse matrix containing reservoir weights
    """
    rnd0, rnd1 = jax.random.split(jax.random.PRNGKey(W_seeds[0]), 2)
    W = jnp.zeros(W_shape)
    # Generate random connections matrix
    W_connection = jax.random.uniform(rnd0, shape=W_shape, minval=0, maxval=1.0)
    # generate the weights from the uniform distribution (-1,1)
    W_weights = jax.random.uniform(rnd1, shape=W_shape, minval=-1.0, maxval=1.0)
    # Apply sparseness condition to generate final weights matrix
    W = jnp.where(W_connection < (1 - sparseness), W_weights, W)
    # Normalize the matrix to control spectral radius
    rho_pre = jnp.abs(jax.scipy.linalg.eigh(W, eigvals_only=True))[0]
    W = (1 / rho_pre) * W
    # W = sparse.csr_fromdense(W)
    return W


def erdos_renyi2(W_shape, sparseness, W_seeds):
    """
    Create the reservoir weights matrix according to Erdos-Renyi network.

    Args:
        W_shape (tuple): Shape of the matrix (rows, columns)
        sparseness (float): Sparseness parameter
        W_seeds (list): A list of seeds for the random generators; one for the connections, one for uniform sampling of weights

    Returns:
        W: sparse matrix containing reservoir weights
    """
    prob = 1 - sparseness
    rnd0, rnd1 = jax.random.split(jax.random.PRNGKey(W_seeds[0]), 2)
    i, j = jnp.indices(W_shape)
    # Generate random values
    b_values = jax.random.uniform(rnd0, shape=W_shape, minval=0.0, maxval=1.0)
    rnd_values = jax.random.uniform(rnd1, shape=W_shape, minval=0.0, maxval=1.0)
    # Mask indices where i != j and b < prob
    mask = (i != j) & (b_values < prob)
    # Set values in the matrix at the specified indices
    W = jnp.zeros(W_shape)
    W = W.at[mask].set(rnd_values[mask])
    # Normalize the matrix to control spectral radius
    rho_pre = jnp.abs(jax.scipy.linalg.eigh(W, eigvals_only=True))[0]
    W = (1 / rho_pre) * W
    # W = sparse.csr_fromdense(W)
    return W
