import jax
import jax.numpy as jnp

from jax import config

config.update("jax_enable_x64", True)


def calculate_LEs_jax(esn_jacobian_jax, reservoir, N_transient, dt, norm_step=1, target_dim=None, randomseed=0):
    """Calculate the Lyapunov exponents
    Args:
        reservoir: state trajectory
        target_dim: dimension of the target system, valid for ESN, e.g. <3 for Lorenz 64
        transient_time: number of transient time steps
        dt: delta t of data
    Returns:
        LEs: Lyapunov exponents
    """
    # total number of time steps
    N = reservoir.shape[0]
    # number of transient steps that will be discarded
    # number of qr normalization steps
    N_qr = int(jnp.ceil((N - N_transient) / norm_step))
    T = jnp.arange(1, N_qr + 1) * dt * norm_step

    # dimension of the system
    dim = reservoir.shape[1]
    if target_dim is None:
        target_dim = dim

    # set random orthonormal Lyapunov vectors (GSVs)
    key = jax.random.PRNGKey(randomseed)
    U = jnp.linalg.qr(jax.random.normal(key, (dim, target_dim)))[0]
    Q, R = jnp.linalg.qr(U)
    U = Q[:, :target_dim]

    def var_equation(n, idx, U):
        def body(carry, _):
            idx, U_current = carry
            jac = jnp.multiply(esn_jacobian_jax, 1.0 - reservoir[idx] ** 2)
            U_new = jnp.matmul(jac, U_current)
            U_new, R = jax.scipy.linalg.qr(U_new[:, :target_dim], mode="economic")
            idx = idx + 1
            return (idx, U_new), (U_new, R, jnp.abs(jnp.diag(R[:target_dim, :target_dim])))

        (idx, U), (U_all, R_all, LE_all) = jax.lax.scan(body, (idx, U), xs=None, length=n)
        return idx, U, (U_all, R_all, LE_all)

    idx, U, _ = var_equation(N_transient, 0, U)
    idx, U, (U_tracked, R_tracked, LEs_tracked) = var_equation(N - N_transient, idx, U)

    LEs = jnp.cumsum(jnp.log(LEs_tracked[:]), axis=0) / jnp.tile(T[:], (target_dim, 1)).T

    # idx, U, (U_tracked, R_tracked, LEs_tracked) = var_equation(N, 0, U)
    # LEs = jnp.cumsum(jnp.log(LEs_tracked[N_transient:]), axis=0) / jnp.tile(T[:], (target_dim, 1)).T
    return LEs
