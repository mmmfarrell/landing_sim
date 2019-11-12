import pytest
import numpy as np
import numdifftools as nd

XGLOBAL = np.random.rand(4, 1)
XGLOBAL /= np.linalg.norm(XGLOBAL)

def err_quat(dx):
    q_tilde = np.zeros((4, 1))
    q_tilde[0] = 1.
    q_tilde[1:] = 0.5 * dx
    q_tilde /= np.linalg.norm(XGLOBAL)
    return q_tilde

def skew(vec):
    skew = np.zeros((3, 3))
    skew[0, 1] = -vec[2]
    skew[0, 2] = vec[1]

    skew[1, 0] = vec[2]
    skew[1, 2] = -vec[0]

    skew[2, 0] = -vec[1]
    skew[2, 1] = vec[0]

    return skew

def otimes(qa, qb):
    qa0 = qa[0]
    qa1 = qa[1]
    qa2 = qa[2]
    qa3 = qa[3]
    qabar = qa[1:]

    qb0 = qb[0]
    qb1 = qb[1]
    qb2 = qb[2]
    qb3 = qb[3]
    qbbar = qb[1:]

    #  mat1 = np.zeros((4, 4))
    #  mat1[0, 0] = -qa0
    #  mat1[0, 1:4] = -qabar.transpose()
    #  mat1[1:4, 0] = np.reshape(qabar, (3))
    #  mat1[1:4, 1:4] = qa0 * np.eye(3) + skew(qabar)
    #  print(skew(qabar))
    #  print(mat1)
    result = np.zeros((4, 1))
    result[0] = qb0 * qa0 - qb1 * qa1 - qb2 * qb2 - qb3 * qa3
    result[1] = qb0 * qa1 + qb1 * qa0 - qb2 * qb3 + qb3 * qa2
    result[2] = qb0 * qa2 + qb1 * qa3 + qb2 * qb0 - qb3 * qa1
    result[3] = qb0 * qa3 - qb1 * qa2 + qb2 * qb1 + qb3 * qa0

    result /= np.linalg.norm(result)
    return result

    #  return np.matmul(mat1, qb)

def box_plus(dx):
    #  print(err_quat(dx))
    return otimes(XGLOBAL, err_quat(dx))

def log(q):
    return 2. * np.sign(q[0]) * q[1:4]

def att_meas_model(x):
    return x

def residual(dx):
    x_est = XGLOBAL
    x_true = box_plus(dx)
    print('x', x_est)
    print('x_true', x_true)
    #  print(x_true)

    h_est = att_meas_model(x_est)
    h_true = att_meas_model(x_true)

    h_est_inv = h_est
    h_est_inv[1:4] = -1 * h_est_inv[1:4]
    print('h inv', h_est_inv)
    print('otimes', otimes(h_est_inv, h_est))
    res = log(otimes(h_est_inv, h_true))
    return res

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
    

    #  jac = jac_func(dx)
    #  analytical_jac = analytical_pos_meas_jac()

    #  res = np.isclose(jac, analytical_jac)
    #  print('jac', jac)
    #  print('ana jac', analytical_jac)
    #  assert(np.all(res)), res


if __name__ == '__main__':
    test_pos_model_jacobian()
