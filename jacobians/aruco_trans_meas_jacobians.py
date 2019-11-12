from __future__ import print_function

import pytest
import numpy as np
import numdifftools as nd

from quat import Quat

XGLOBAL = np.random.rand(4, 1)
XGLOBAL /= np.linalg.norm(XGLOBAL)

class State:
    def __init__(self):
        pass



def att_meas_model(x):
    return x

def residual(dx):
    pass
    #  x_est = XGLOBAL
    #  x_true = box_plus(dx)
    #  print('x', x_est)
    #  print('x_true', x_true)
    #  #  print(x_true)

    #  h_est = att_meas_model(x_est)
    #  h_true = att_meas_model(x_true)

    #  h_est_inv = h_est
    #  h_est_inv[1:4] = -1 * h_est_inv[1:4]
    #  print('h inv', h_est_inv)
    #  print('otimes', otimes(h_est_inv, h_est))
    #  res = log(otimes(h_est_inv, h_true))
    #  return res

    #  return h_true - h_est

def analytical_pos_meas_jac():
    #  return np.zeros_like(XGLOBAL)
    return np.zeros((3, 3))

def test_pos_model_jacobian():
    #  dx = 1e-3 * np.random.rand(3, 1)
    dx = np.zeros((3, 1))
    jac_func = nd.Jacobian(residual)
    r = residual(dx)
    print('res', r)

    q = Quat()
    print('q', q)
    

    #  jac = jac_func(dx)
    #  analytical_jac = analytical_pos_meas_jac()

    #  res = np.isclose(jac, analytical_jac)
    #  print('jac', jac)
    #  print('ana jac', analytical_jac)
    #  assert(np.all(res)), res


if __name__ == '__main__':
    test_pos_model_jacobian()
