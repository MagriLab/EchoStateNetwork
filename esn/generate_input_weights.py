# Input weights generation methods
import numpy as np
from scipy.sparse import lil_matrix


def sparse_random(W_in_shape, W_in_seeds):
    """Create the input weights matrix
    Inputs are not connected and the weights are randomly placed one per row.

    Args:
        W_in_shape: N_reservoir x (N_inputs + N_input_bias)
        seeds: a list of seeds for the random generators;
            one for the column index, one for the uniform sampling
    Returns:
        W_in: sparse matrix containing the input weights
    """
    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    # set the seeds
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])
    rnd2 = np.random.RandomState(W_in_seeds[2])

    # make W_in
    for j in range(W_in_shape[0]):
        rnd_idx = rnd0.randint(0, W_in_shape[1])
        # only one element different from zero
        # sample from the uniform distribution
        W_in[j, rnd_idx] = rnd1.uniform(-1, 1)

    # input associated with system's bifurcation parameters are
    # fully connected to the reservoir states

    W_in = W_in.tocsr()

    return W_in

def sparse_random_dense_input_bias(W_in_shape, W_in_seeds, input_bias_len):
    """Create the input weights matrix.
    Inputs are not connected and the weights are randomly placed one per row
    However, the input bias is densely connected to ALL reservoir state variables

    Args:
        W_in_shape: N_reservoir x (N_inputs + N_input_bias)
        W_in_seeds: a list of seeds for the random generators;
            one for the column index, one for the uniform sampling
    Returns:
        W_in: sparse matrix containing the input weights
    """
    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    # set the seeds
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    # make W_in
    n_cols = W_in_shape[1] - input_bias_len
    for j in range(W_in_shape[0]):
        rnd_idx = rnd0.randint(0, n_cols)  # low inclusive, high exclusive
        # only one element different from zero
        # sample from the uniform distribution
        W_in[j, rnd_idx] = rnd1.uniform(-1, 1)

    # input bias is fully connected to the reservoir states
    if input_bias_len > 0:
        W_in[:, W_in_shape[1] - input_bias_len:] = rnd1.uniform(
            -1, 1, (W_in_shape[0], input_bias_len)
        )

    W_in = W_in.tocsr()

    return W_in

def sparse_grouped(W_in_shape, W_in_seeds):
    # The inputs are not connected but they are grouped within the matrix

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    for i in range(W_in_shape[0]):
        W_in[
            i,
            int(np.floor(i * (W_in_shape[1]) / W_in_shape[0])),
        ] = rnd0.uniform(-1, 1)

    W_in = W_in.tocsr()
    return W_in

def sparse_grouped_dense_input_bias(W_in_shape, W_in_seeds, input_bias_len):
    # The inputs are not connected but they are grouped within the matrix

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])

    n_cols = W_in_shape[1] - input_bias_len
    for i in range(W_in_shape[0]):
        W_in[
            i,
            int(np.floor(i * n_cols / W_in_shape[0])),
        ] = rnd0.uniform(-1, 1)

    # input bias is fully connected to the reservoir states
    if input_bias_len > 0:
        W_in[:, W_in_shape[1] - input_bias_len:] = rnd0.uniform(
            -1, 1, (W_in_shape[0], input_bias_len)
        )
    W_in = W_in.tocsr()
    return W_in

def dense(W_in_shape, W_in_seeds):
    # The inputs are all connected

    rnd0 = np.random.RandomState(W_in_seeds[0])
    W_in = rnd0.uniform(-1, 1, W_in_shape)
    return W_in
