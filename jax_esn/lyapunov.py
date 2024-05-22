import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import qr



def ESN_variation(sys, u, u_prev, M):
    """Variation of the ESN.
    Evolution in the tangent space.
    """
    dtanh_val = sys.dtanh(u, u_prev)[:, None]
    # jacobian of the reservoir dynamics
    jac_val = sys.jac(dtanh_val, u_prev)
    M_next = jnp.matmul(jac_val, M)  # because ESN discrete time map
    return M_next


def calculate_LEs_less_storage(
    sys, sys_type, X, t, N_transient, dt, norm_step=1, target_dim=None
):
    """Calculate the Lyapunov exponents but doesn't save Q or R
    Args:
        sys: system object that contains the governing equations and jacobian
        sys_type: whether system is continuous time or an ESN
        X: state trajectory if continuous or reservoir if ESN
        t: time
        target_dim: dimension of the target system, valid for ESN
        dt: time steps
    Returns:
        LEs: Lyapunov exponents
        QQ, RR can be used for the computation of Covariant Lyapunov Vectors
    """
    # total number of time steps
    N = X.shape[0]
    # number of transient steps that will be discarded
    # number of qr normalization steps
    N_qr = int(jnp.ceil((N - 1 - N_transient) / norm_step))
    T = jnp.arange(1, N_qr + 1) * dt * norm_step

    # dimension of the system
    dim = X.shape[1]
    if target_dim is None:
        target_dim = dim

    # Lyapunov Exponents timeseries
    LE = jnp.zeros((N_qr, target_dim))
    # finite-time Lyapunov Exponents timeseries
    FTLE = jnp.zeros((N_qr, target_dim))
    # set random orthonormal Lyapunov vectors (GSVs)
    key = random.PRNGKey(0)
    U = jnp.linalg.qr(random.normal(key, (dim, target_dim)))[0]
    Q, R =  jnp.linalg.qr(U)
    U = Q[:, :target_dim]

    idx = 0
    for i in range(1, N):
        U = ESN_variation(sys, X[i], X[i - 1], U)

        if i % norm_step == 0:
            Q, R =  jnp.linalg.qr(U)
            U = Q[:, :target_dim]
            if i > N_transient:
                LE = LE.at[idx].set(jnp.abs(jnp.diag(R[:target_dim, :target_dim])))
                FTLE = FTLE.at[idx].set((1.0 / dt) * jnp.log(LE[idx]))
                idx += 1

    LEs = jnp.cumsum(jnp.log(LE[:]), axis=0) / jnp.tile(T[:], (target_dim, 1)).T
    return LEs, FTLE
