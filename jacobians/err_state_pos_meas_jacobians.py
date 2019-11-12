import pytest
import numpy as np
import numdifftools as nd

XGLOBAL = np.random.rand(3, 1)

def box_plus(dx):
    return XGLOBAL + dx

def pos_meas_model(x):
    return x

def residual(dx):
    x_est = XGLOBAL
    x_true = box_plus(dx)

    h_est = pos_meas_model(x_est)
    h_true = pos_meas_model(x_true)

    return h_true - h_est

def analytical_pos_meas_jac(x):
    return np.zeros_like(x)

def test_pos_model_jacobian():
    dx = np.random.rand(3, 1)
    jac_func = nd.Jacobian(residual)

    jac = jac_func(dx)
    analytical_jac = analytical_pos_meas_jac()

    res = np.isclose(jac, analytical_jac)
    assert(np.all(res)), res


if __name__ == '__main__':
    test_pos_model_jacobian()
