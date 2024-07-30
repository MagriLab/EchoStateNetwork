from itertools import combinations

import numpy as np
import scipy


def qr_factorization(A):
    """QR Decomposition using Gram-Schmidt process
    Decompose A = QR s.t.
    Q is orthogonal (columns are orthogonal unit vectors Q^T = Q^-1)
    R is upper triangular
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            q = Q[:, i]
            R[i, j] = q.dot(v)
            v = v - R[i, j] * q

        norm = np.linalg.norm(v)
        Q[:, j] = v / norm
        R[j, j] = norm
    return Q, R


def normalize(A):
    """Normalize columns of A"""
    norm = scipy.linalg.norm(A, axis=0)
    return A / norm, norm


def continuous_tangent_step(ddt, jac, u, M, t):
    """Evolution in the tangent space
    Args:
        ddt: governing ODEs, dudt = F(u, t)
        jac: jacobian, dF/du
        M: perturbations in the tangent space
    """
    dudt = ddt(u, t)
    dMdt = np.dot(jac(u), M)
    return dudt, dMdt


def rk4var_step(ddt, jac, u, M, t, dt, *args):
    """Variational step with RK4 method"""
    K1, M_K1 = continuous_tangent_step(ddt, jac, u, M, t)
    K2, M_K2 = continuous_tangent_step(ddt, jac, u + dt * K1 / 2.0, M + dt * M_K1 / 2.0, t + dt / 2.0)
    K3, M_K3 = continuous_tangent_step(ddt, jac, u + dt * K2 / 2.0, M + dt * M_K2 / 2.0, t + dt / 2.0)
    K4, M_K4 = continuous_tangent_step(ddt, jac, u + dt * K3, M + dt * M_K3, t + dt)

    du = dt * (K1 / 2.0 + K2 + K3 + K4 / 2.0) / 3.0
    dM = dt * (M_K1 / 2.0 + M_K2 + M_K3 + M_K4 / 2.0) / 3.0
    return du, dM


def continuous_variation(sys, u, M, t, dt):
    """Variation for continuous time systems."""
    _, dM = rk4var_step(sys.ode, sys.jac, u, M, t, dt)
    M_next = M + dM
    return M_next


def ESN_variation(sys, u, u_prev, M):
    """Variation of the ESN.
    Evolution in the tangent space.
    """
    dtanh = sys.dtanh(u, u_prev)[:, None]
    # jacobian of the reservoir dynamics
    jac = sys.jac(dtanh, u_prev)
    M_next = np.matmul(jac, M)  # because ESN discrete time map
    return M_next


def calculate_LEs_less_storage(sys, sys_type, X, t, N_transient, dt, norm_step=1, target_dim=None):
    """Calculate the Lyapunov exponents
    Args:
        sys: system object that contains the governing equations and jacobian
        sys_type: whether system is continuous time or an ESN
        X: state trajectory
        t: time
        target_dim: dimension of the target system, valid for ESN
        transient_time: number of transient time steps
        dt: time steps
    Returns:
        LEs: Lyapunov exponents
        QQ: Q matrix recorded in time
        RR: R matrix recorded in time
        QQ, RR can be used for the computation of Covariant Lyapunov Vectors
    """
    # total number of time steps
    N = X.shape[0]
    # number of transient steps that will be discarded
    # number of qr normalization steps
    N_qr = int(np.ceil((N - 1 - N_transient) / norm_step))
    T = np.arange(1, N_qr + 1) * dt * norm_step

    # dimension of the system
    dim = X.shape[1]
    if target_dim is None:
        target_dim = dim

    # Lyapunov Exponents timeseries
    LE = np.zeros((N_qr, target_dim))
    # finite-time Lyapunov Exponents timeseries
    FTLE = np.zeros((N_qr, target_dim))
    # set random orthonormal Lyapunov vectors (GSVs)
    U = scipy.linalg.orth(np.random.rand(dim, target_dim))
    Q, R = qr_factorization(U)
    U = Q[:, :target_dim]

    idx = 0
    for i in range(1, N):
        if sys_type == "continuous":
            U = continuous_variation(sys, X[i], U, t[i], dt)
        elif sys_type == "ESN":
            U = ESN_variation(sys, X[i], X[i - 1], U)

        if i % norm_step == 0:
            Q, R = qr_factorization(U)
            U = Q[:, :target_dim].copy()
            if i > N_transient:
                LE[idx] = np.abs(np.diag(R[:target_dim, :target_dim]))
                FTLE[idx] = (1.0 / dt) * np.log(LE[idx])
                idx += 1

    LEs = np.cumsum(np.log(LE[:]), axis=0) / np.tile(T[:], (target_dim, 1)).T
    return LEs, FTLE


def calculate_LEs(sys, sys_type, X, t, N_transient, dt, norm_step=1, target_dim=None):
    """Calculate the Lyapunov exponents
    Args:
        sys: system object that contains the governing equations and jacobian
        sys_type: whether system is continuous time or an ESN
        X: state trajectory
        t: time
        target_dim: dimension of the target system, valid for ESN
        transient_time: number of transient time steps
        dt: time steps
    Returns:
        LEs: Lyapunov exponents
        QQ: Q matrix recorded in time
        RR: R matrix recorded in time
        QQ, RR can be used for the computation of Covariant Lyapunov Vectors
    """
    # total number of time steps
    N = X.shape[0]
    # number of transient steps that will be discarded
    # number of qr normalization steps
    N_qr = int(np.ceil((N - 1 - N_transient) / norm_step))
    T = np.arange(1, N_qr + 1) * dt * norm_step

    # dimension of the system
    dim = X.shape[1]
    if target_dim is None:
        target_dim = dim

    # Lyapunov Exponents timeseries
    LE = np.zeros((N_qr, target_dim))
    # finite-time Lyapunov Exponents timeseries
    FTLE = np.zeros((N_qr, target_dim))
    # Q matrix recorded in time
    QQ = np.zeros((dim, target_dim, N_qr))
    # R matrix recorded in time
    RR = np.zeros((target_dim, target_dim, N_qr))

    # set random orthonormal Lyapunov vectors (GSVs)
    U = scipy.linalg.orth(np.random.rand(dim, target_dim))
    Q, R = qr_factorization(U)
    U = Q[:, :target_dim]

    idx = 0
    for i in range(1, N):
        if sys_type == "continuous":
            U = continuous_variation(sys, X[i], U, t[i], dt)
        elif sys_type == "ESN":
            U = ESN_variation(sys, X[i], X[i - 1], U)

        if i % norm_step == 0:
            Q, R = qr_factorization(U)
            U = Q[:, :target_dim].copy()
            if i > N_transient:
                QQ[:, :, idx] = Q.copy()
                RR[:, :, idx] = R.copy()
                LE[idx] = np.abs(np.diag(R[:target_dim, :target_dim]))
                FTLE[idx] = (1.0 / dt) * np.log(LE[idx])
                idx += 1

    LEs = np.cumsum(np.log(LE[:]), axis=0) / np.tile(T[:], (target_dim, 1)).T
    return LEs, FTLE, QQ, RR


def timeseriesdot(x, y, multype):
    tsdot = np.einsum(multype, x, y.T)  # Einstein summation. Index i is time.
    return tsdot


def CLV_angles(clv, target_dim):
    # calculate angles between CLVs
    costhetas = np.zeros((clv[:, 0, :].shape[1], target_dim))
    count = 0
    for subset in combinations(np.arange(target_dim), target_dim - 1):
        # each column is a combination of the dimensions (0,1),(0,2),(1,2)...etc
        index1 = subset[0]
        index2 = subset[1]

        # For principal angles take the absolute of the dot product
        # take dot product of CLVs at each time step and collect
        costhetas[:, count] = np.absolute(timeseriesdot(clv[:, index1, :], clv[:, index2, :], "ij,ji->j"))
        count += 1
    thetas = 180.0 * np.arccos(costhetas) / np.pi

    return thetas


def calculate_CLVs(QQ, RR, dt):
    """Calculate covariant lyapunov vectors (CLVs)
    Args:
        QQ: Q matrix collected in time
        RR: R matrix collected in time
        dt: time step
    Returns:
        V: CLVs
        theta: angles between CLVs
        ftcle: finite-time lyapunov exponents along CLVs
    """
    N = QQ.shape[2]
    su = int(N / 10)  # spinup time
    sd = int(N / 10)  # spindown time
    s = su  # index of spinup time
    e = N + 1 - sd  # index of spindown time

    # initialise components
    dim = QQ.shape[0]
    target_dim = QQ.shape[1]

    C = np.zeros((target_dim, target_dim, N))  # coordinates of CLVs in local GS vector basis
    D = np.zeros((target_dim, N))  # diagonal matrix with CLV growth factors
    V = np.zeros((dim, target_dim, N))  # coordinates of CLVs in physical space (each column is a vector)

    C[:, :, -1] = np.eye(target_dim)
    D[:, -1] = np.ones(target_dim)
    V[:, :, -1] = np.dot(np.real(QQ[:, :, -1]), C[:, :, -1])

    # integrate backwards
    for i in reversed(range(N - 1)):
        C[:, :, i], D[:, i] = normalize(scipy.linalg.solve_triangular(np.real(RR[:, :, i]), C[:, :, i + 1]))
        V[:, :, i] = np.dot(np.real(QQ[:, :, i]), C[:, :, i])

    # normalize CLVs
    V, _ = normalize(V)
    # find the angles
    theta = CLV_angles(V, target_dim)

    # FTCLE: Finite-time lyapunov exponents along CLVs
    ftcle = np.zeros((target_dim, N + 1))

    # compute the FTCLEs
    for j in 1 + np.arange(s, e):  # time loop
        ftcle[:, j] = -(1.0 / dt) * np.log(D[:, j])

    return V, theta, ftcle
