import numpy as np


def ddt(u, params):
    beta, rho, sigma = params
    x, y, z = u
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])


def l63jac(u, params):
    beta, rho, sigma = params  # Unpack the constants vector
    x, y, z = u  # Unpack the state vector

    # Jacobian
    J = np.array([[-sigma, sigma,     0],
                  [rho-z,    -1,    -x],
                  [y,     x, -beta]])
    return J


def RK4(ddt, u0, T, params):
    u = np.empty((T.size, u0.size))
    u[0] = u0
    der = np.empty((T.size, u0.size))
    for i in range(1, T.size):
        delta_t = (T[i] - T[i-1])
        K1 = ddt(u[i-1], params)
        K2 = ddt(u[i-1] + delta_t*K1/2.0, params)
        K3 = ddt(u[i-1] + delta_t*K2/2.0, params)
        K4 = ddt(u[i-1] + delta_t*K3, params)
        u[i:] = u[i-1] + np.array(delta_t * (K1/2.0 + K2 + K3 + K4/2.0) / 3.0)
        der[i-1] = K1
    return u, der


def solve_l63_rk4(N, dt, u0, params):
    T = np.arange(N+1) * dt
    U, derU = RK4(ddt, u0, T, params)
    return T, U, derU
