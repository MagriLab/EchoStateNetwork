import jax
import jax.numpy as jnp
from jax_esn import generate_input_weights, generate_reservoir_weights


class ESN:
    def __init__(
        self,
        reservoir_size,
        dimension,
        reservoir_connectivity=0,
        input_normalization=None,
        input_scaling=1.0,
        tikhonov=1e-9,
        spectral_radius=1.0,
        leak_factor=1.0,
        input_bias=jnp.array([]),
        output_bias=jnp.array([]),
        input_seeds=[1, 2, 3],
        reservoir_seeds=[1, 2],
        verbose=True,
        input_weights_mode="sparse_random",
        reservoir_weights_mode="erdos_renyi2",
    ):
        """Creates an Echo State Network with the given parameters
        Args:
            reservoir_size: number of neurons in the reservoir
            dimension: dimension of the state space of the input and output
                they must have the same size in order for the closed-loop to work
            reservoir_connectivity: connectivity of the reservoir weights,
                how many connections does each neuron have (on average)
            input_normalization: normalization applied to the input before activation
                tuple with (mean, norm) such that input u is updated as (u-mean)/norm
            input_scaling: scaling applied to the input weights matrix
            spectral_radius: spectral radius (maximum absolute eigenvalue)
                of the reservoir weights matrix
            leak_factor: factor for the leaky integrator
                if set to 1 (default), then no leak is applied
            input_bias: bias that is augmented to the input vector
            output_bias: bias that is augmented to the reservoir state vector before readout
            input_seeds: seeds to generate input weights matrix
            reservoir_seeds: seeds to generate reservoir weights matrix
        Returns:
            ESN object

        """
        self.verbose = verbose

        # Hyperparameters

        # the reservoir size and dimension should be fixed during initialization and not changed since they affect
        # the matrix dimensions, and the matrices can become incompatible
        self.N_reservoir = reservoir_size
        self.N_dim = dimension

        # Leak factor
        self.leak_factor = leak_factor

        # Biases
        self.input_bias = input_bias
        self.output_bias = output_bias

        # Input normalization
        if not input_normalization:
            input_normalization = [None] * 2
            input_normalization[0] = jnp.zeros(self.N_dim)
            input_normalization[1] = jnp.ones(self.N_dim)

        self.input_normalization = input_normalization

        # Weights
        # the object should also store the seeds for reproduction
        # initialise input weights
        self.W_in_seeds = input_seeds
        self.W_in_shape = (self.N_reservoir, self.N_dim + len(self.input_bias))
        # N_dim+length of input bias because we augment the inputs with a bias
        # if no bias, then this will be + 0
        self.input_weights_mode = input_weights_mode
        self.input_weights = self.generate_input_weights()
        self.input_scaling = input_scaling
        self.tikh = tikhonov
        # input weights are automatically scaled if input scaling is updated

        self.reservoir_connectivity = reservoir_connectivity
        self.W_seeds = reservoir_seeds
        self.W_shape = (self.N_reservoir, self.N_reservoir)
        self.reservoir_weights_mode = reservoir_weights_mode
        self.reservoir_weights = self.generate_reservoir_weights()
        self.spectral_radius = spectral_radius
        # reservoir weights are automatically scaled if spectral radius is updated

        # initialise output weights
        self.W_out_shape = (self.N_reservoir + len(self.output_bias), self.N_dim)
        # N_reservoir+length of output bias because we augment the outputs with a bias
        # if no bias, then this will be + 0
        self.output_weights = jnp.zeros(self.W_out_shape)

        # self._dfdu_const = None
        # self._dudx_const = None
        # self._dfdu_dudx_const = None

        # self.step = jax.jit(jax.tree_util.Partial(self._step,TRAINING=False))

        # self.step_jit = jax.jit(jax.tree_util.Partial(step, [jnp.array(self.norm_in), self.b_in, self.W_in, self.W, self.alpha]))

    @property
    def reservoir_connectivity(self):
        return self.connectivity

    @reservoir_connectivity.setter
    def reservoir_connectivity(self, new_reservoir_connectivity):
        # set connectivity
        if new_reservoir_connectivity <= 0:
            raise ValueError("Connectivity must be greater than 0.")
        self.connectivity = new_reservoir_connectivity
        # regenerate reservoir with the new connectivity
        if hasattr(self, "W"):
            if self.verbose:
                print("Reservoir weights are regenerated for the new connectivity.")
            self.reservoir_weights = self.generate_reservoir_weights()
        return

    @property
    def leak_factor(self):
        return self.alpha

    @leak_factor.setter
    def leak_factor(self, new_leak_factor):
        # set leak factor
        if new_leak_factor < 0 or new_leak_factor > 1:
            raise ValueError("Leak factor must be between 0 and 1 (including).")
        self.alpha = new_leak_factor

    @property
    def tikhonov(self):
        return self.tikh

    @tikhonov.setter
    def tikhonov(self, new_tikhonov):
        # set tikhonov coefficient
        if new_tikhonov <= 0:
            raise ValueError("Tikhonov coefficient must be greater than 0.")
        self.tikh = new_tikhonov

    @property
    def input_normalization(self):
        return self.norm_in

    @input_normalization.setter
    def input_normalization(self, new_input_normalization):
        self.norm_in = new_input_normalization
        if self.verbose:
            print("Input normalization is changed, training must be done again.")

    @property
    def input_scaling(self):
        return self.sigma_in

    @input_scaling.setter
    def input_scaling(self, new_input_scaling):
        """Setter for the input scaling, if new input scaling is given,
        then the input weight matrix is also updated
        """
        if hasattr(self, "sigma_in"):
            # rescale the input matrix
            self.W_in = (1 / self.sigma_in) * self.W_in  # .todense()
            # self.W_in = sparse.csr_fromdense(self.W_in)
        # set input scaling

        if self.verbose:
            print("Input weights are rescaled with the new input scaling.")
        self.sigma_in = new_input_scaling
        self.W_in = self.sigma_in * self.W_in  # .todense()
        # self.W_in = sparse.csr_fromdense(self.W_in)
        return

    @property
    def spectral_radius(self):
        return self.rho

    @spectral_radius.setter
    def spectral_radius(self, new_spectral_radius):
        """Setter for the spectral_radius, if new spectral_radius is given,
        then the reservoir weight matrix is also updated
        """
        if hasattr(self, "rho"):
            # rescale the reservoir matrix

            self.W = (1 / self.rho) * self.W  # .todense()
            # self.W = sparse.csr_fromdense(self.W)
        # set spectral radius

        if self.verbose:
            print("Reservoir weights are rescaled with the new spectral radius.")
        self.rho = new_spectral_radius
        self.W = self.rho * self.W
        # self.W = sparse.csr_fromdense(self.W)
        return

    @property
    def input_weights(self):
        return self.W_in

    @input_weights.setter
    def input_weights(self, new_input_weights):
        # first check the dimensions
        if new_input_weights.shape != self.W_in_shape:
            raise ValueError(
                f"The shape of the provided input weights does not match with the network, {new_input_weights.shape} != {self.W_in_shape}"
            )

        # set the new input weights
        self.W_in = new_input_weights

        # set the input scaling to 1.0
        if self.verbose:
            print("Input scaling is set to 1, set it separately if necessary.")
        self.sigma_in = 1.0
        return

    @property
    def reservoir_weights(self):
        return self.W

    @reservoir_weights.setter
    def reservoir_weights(self, new_reservoir_weights):
        # first check the dimensions
        if new_reservoir_weights.shape != self.W_shape:
            raise ValueError(
                f"The shape of the provided reservoir weights does not match with the network," f"{new_reservoir_weights.shape} != {self.W_shape}"
            )

        # set the new reservoir weights
        self.W = new_reservoir_weights

        # set the spectral radius to 1.0
        if self.verbose:
            print("Spectral radius is set to 1, set it separately if necessary.")
        self.rho = 1.0
        return

    @property
    def output_weights(self):
        return self.W_out

    @output_weights.setter
    def output_weights(self, new_output_weights):
        # first check the dimensions
        if new_output_weights.shape != self.W_out_shape:
            raise ValueError(
                f"The shape of the provided output weights does not match with the network," f"{new_output_weights.shape} != {self.W_out_shape}"
            )
        # set the new reservoir weights
        self.W_out = new_output_weights
        return

    @property
    def input_bias(self):
        return self.b_in

    @input_bias.setter
    def input_bias(self, new_input_bias):
        self.b_in = new_input_bias
        return

    @property
    def output_bias(self):
        return self.b_out

    @output_bias.setter
    def output_bias(self, new_output_bias):
        self.b_out = new_output_bias
        return

    @property
    def sparseness(self):
        """Define sparseness from connectivity"""
        # probability of non-connections = 1 - probability of connection
        # probability of connection = (number of connections)/(total number of neurons - 1)
        # -1 to exclude the neuron itself
        return 1 - (self.connectivity / (self.N_reservoir - 1))

    def generate_input_weights(self):
        if self.input_weights_mode == "sparse_random":
            return generate_input_weights.sparse_random(self.W_in_shape, self.W_in_seeds)
        elif self.input_weights_mode == "sparse_grouped":
            return generate_input_weights.sparse_grouped(self.W_in_shape, self.W_in_seeds)
        elif self.input_weights_mode == "dense":
            return generate_input_weights.dense(self.W_in_shape, self.W_in_seeds)
        else:
            raise ValueError("Not valid input weights generator.")

    def generate_reservoir_weights(self):
        if self.reservoir_weights_mode == "erdos_renyi1":
            return generate_reservoir_weights.erdos_renyi1(self.W_shape, self.sparseness, self.W_seeds)
        if self.reservoir_weights_mode == "erdos_renyi2":
            return generate_reservoir_weights.erdos_renyi2(self.W_shape, self.sparseness, self.W_seeds)
        else:
            raise ValueError("Not valid reservoir weights generator.")

    def calculate_constant_jacobian(self):
        dfdx_x = self.W
        # gradient of x(i+1) with x(i) due to u(i) (in closed-loop)
        dfdx_u = (self.W_in / self.norm_in[1]) @ self.W_out[: self.N_reservoir, :].T
        return dfdx_x + dfdx_u


def step(params, x_prev, u):
    # donate args
    """Advances ESN time step.
    Args:
        x_prev: reservoir state in the previous time step (n-1)
        u: input in this time step (n)
    Returns:
        x: reservoir state in this time step (n)
    """
    # normalise the input
    u_norm = (u - params.norm_in[0]) / params.norm_in[1]
    # we normalize here, so that the input is normalised
    # in closed-loop run too

    # augment the input with the input bias
    u_augmented = jnp.hstack((u_norm, params.b_in))

    # update the reservoir
    x_tilde = jnp.tanh(jnp.dot(params.W_in, u_augmented) + jnp.dot(params.W, x_prev))

    # apply the leaky integrator
    x = (1 - params.alpha) * x_prev + params.alpha * x_tilde
    return x


def open_loop(params, x0, U):
    """Advances ESN in open-loop.
    Args:
        x0: initial reservoir state
        U: input time series in matrix form (N_t x N_dim)

    Returns:
        X: time series of the reservoir states (N_t x N_reservoir)
    """

    # we write a bodyfunction to apply jax.lax.scan
    # which implements the for loop
    def fn_body(x_prev, u):
        x = step(params, x_prev, u)
        return x, x

    x_final, X_preceed = jax.lax.scan(fn_body, x0, U, None)
    X = jnp.vstack((x0, X_preceed))
    return x_final, X


def run_washout(params, U_washout):
    # Wash-out phase to get rid of the effects of reservoir states initialised as zero
    # initialise the reservoir states before washout
    x0_washout = jnp.zeros(params.N_reservoir)

    # let the ESN run in open-loop for the wash-out
    # get the initial reservoir to start the actual open/closed-loop,
    # which is the last reservoir state
    x_final, _ = open_loop(params, x0=x0_washout, U=U_washout)
    return x_final


def open_loop_with_washout(params, U_washout, U):
    x0 = run_washout(params, U_washout)
    _, X = open_loop(params, x0=x0, U=U)
    return X


def closed_loop(params, x0, N_t):
    """Advances ESN in closed-loop.
    Args:
        N_t: number of time steps
        x0: initial reservoir state
    Returns:
        X: time series of the reservoir states (N_t x N_reservoir)
        Y: time series of the output (N_t x N_dim)
    """
    x0_augmented = jnp.hstack((x0, params.b_out))
    y0 = jnp.dot(x0_augmented, params.W_out)

    def fn_body(carry, _):
        x_prev, y_prev = carry
        x = step(params, x_prev, y_prev)
        x_augmented = jnp.hstack((x, params.b_out))
        y = jnp.dot(x_augmented, params.W_out)
        return (x, y), (x, y)

    carry0 = (x0, y0)
    (x_final, y_final), (X_preceed, Y_preceed) = jax.lax.scan(fn_body, carry0, None, length=N_t)
    X = jnp.vstack((x0, X_preceed))
    Y = jnp.vstack((y0, Y_preceed))
    return X, Y


def closed_loop_with_washout(params, U_washout, N_t):
    x0 = run_washout(params, U_washout)
    return closed_loop(params, x0=x0, N_t=N_t)


def solve_ridge(X, Y, tikh):
    """Solves the ridge regression problem
    Args:
        X: input data (N_t x (N_reservoir + N_bias))
        Y: output data (N_t x N_dim)
        tikh: weighing tikhonov coefficient that regularises L2 norm
    Output: W_out of size ((N_reservoir+N_bias) x N_dim)
    """

    A = X.T @ X + tikh * jnp.eye(X.shape[1])
    b = X.T @ Y
    return jnp.linalg.solve(A, b)


def train(params, U_washout, U_train, Y_train, tikh):
    """Trains ESN and sets the output weights.
    Args:
        U_washout: washout input time series
        U_train: training input time series
        Y_train: training output time series
        (list of time series if more than one trajectories)
        tikhonov: regularization coefficient
        train_idx_list: if list of time series, then which ones to use in training
            if not specified, all are used
    """
    # get the training input
    # reservoir train is one step longer than U_train and Y_train, we discard the initial state
    X_train = open_loop_with_washout(params, U_washout, U_train)[1:, :]

    # this is the reservoir states augmented with the bias after a washout phase, one bias per time step
    X_train_augmented = jnp.hstack((X_train, params.b_out * jnp.ones((X_train.shape[0], 1))))

    # solve for W_out using linalg solve
    return solve_ridge(X_train_augmented, Y_train, tikh)


def train_mem(params, U_washout, U_train, Y_train, tikh, N_splits):
    # initial washout
    x0 = run_washout(params, U_washout)

    # body function
    # we can split the operations (in timesteps) required for training
    # so we don't need to store a large matrix for X and do the X.T @ X operation
    def fn_body(carry, UY):
        U, Y = UY
        x0, LHS, RHS = carry
        xf, X = open_loop(params, x0, U)
        X_augmented = jnp.hstack((X, params.b_out * jnp.ones((X.shape[0], 1))))
        LHS += X_augmented[1:].T @ X_augmented[1:]
        RHS += X_augmented[1:].T @ Y
        return (xf, LHS, RHS), None

    carry0 = (x0, jnp.zeros((params.N_reservoir, params.N_reservoir)), jnp.zeros((params.N_reservoir, params.N_dim)))
    U_train_split = jnp.reshape(U_train, (N_splits, U_train.shape[0] // N_splits, U_train.shape[1]))
    Y_train_split = jnp.reshape(Y_train, (N_splits, Y_train.shape[0] // N_splits, Y_train.shape[1]))
    (_, LHS, RHS), _ = jax.lax.scan(fn_body, carry0, (U_train_split, Y_train_split), length=N_splits)

    # two options to add the tikhonov coefficient to the diagonal
    # need to test which one is better
    # the second avoids creating a large matrix for tikhonov
    # reg_LHS = LHS + tikh*jnp.eye(LHS.shape[1])
    LHS = jnp.reshape(LHS, -1).at[:: LHS.shape[1] + 1].add(tikh).reshape(*LHS.shape)
    return jnp.linalg.solve(LHS, RHS)

    # def make_step(esn_attr):
    #     return jax.jit(jax.tree_util.Partial(step, esn_attr))

    # def dfdu_const(self):
    #     if self._dfdu_const is None:
    #         try:
    #             self._dfdu_const = self.alpha * self.W_in[:, : self.N_dim] * (1.0 / self.norm_in[1][: self.N_dim])
    #         except:
    #             self._dfdu_const = self.alpha * (self.W_in[:, : self.N_dim] * (1.0 / self.norm_in[1][: self.N_dim]))
    #     return self._dfdu_const

    # def dudx_const(self):
    #     return self.W_out[: self.N_reservoir, :].T

    # def dfdu_dudx_const(self):
    #     if self._dfdu_dudx_const is None:
    #         self._dfdu_dudx_const = jnp.dot(self.dfdu_const(), self.W_out[: self.N_reservoir, :].T)
    #     return self._dfdu_dudx_const

    # def dtanh(self, x, x_prev):
    #     x_tilde = (x - (1 - self.alpha) * x_prev) / self.alpha
    #     dtanh = 1.0 - x_tilde**2
    #     return dtanh

    # def dfdx_u(self, dtanh):
    #     return jnp.multiply(self.dfdu_dudx_const(), dtanh)

    # def jac(self, dtanh, x_prev=None):
    #     dfdx_x = (1 - self.alpha) * jnp.eye(self.N_reservoir) + self.alpha * self.W * dtanh
    #     dfdx = dfdx_x + self.dfdx_u(dtanh)
    #     return dfdx
