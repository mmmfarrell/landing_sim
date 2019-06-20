import pytest
import numpy as np
import numdifftools as nd

GRAV = 9.81

def RotInertial2Body(phi, theta, psi):
    rot = np.zeros((3, 3))

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    rot[0, 0] = ct * cpsi
    rot[0, 1] = ct * spsi
    rot[0, 2] = -st

    rot[1, 0] = sp * st * cpsi - cp * spsi
    rot[1, 1] = sp * st * spsi + cp * cpsi
    rot[1, 2] = sp * ct

    rot[2, 0] = cp * st * cpsi + sp * spsi
    rot[2, 1] = cp * st * spsi - sp * cpsi
    rot[2, 2] = cp * ct

    return rot

def RotI2BdPhi(phi, theta, psi):
    rot = np.zeros((3, 3))

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    rot[0, 0] = 0.
    rot[0, 1] = 0.
    rot[0, 2] = 0.

    rot[1, 0] = cp * st * cpsi + sp * spsi
    rot[1, 1] = cp * st * spsi - sp * cpsi
    rot[1, 2] = cp * ct

    rot[2, 0] = -sp * st * cpsi + cp * spsi
    rot[2, 1] = -sp * st * spsi - cp * cpsi
    rot[2, 2] = -sp * ct

    return rot

def RotI2BdTheta(phi, theta, psi):
    rot = np.zeros((3, 3))

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    rot[0, 0] = -st * cpsi
    rot[0, 1] = -st * spsi
    rot[0, 2] = -ct

    rot[1, 0] = sp * ct * cpsi
    rot[1, 1] = sp * ct * spsi
    rot[1, 2] = sp * -st

    rot[2, 0] = cp * ct * cpsi
    rot[2, 1] = cp * ct * spsi
    rot[2, 2] = cp * -st

    return rot

def RotI2BdPsi(phi, theta, psi):
    rot = np.zeros((3, 3))

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    rot[0, 0] = -ct * spsi
    rot[0, 1] = ct * cpsi
    rot[0, 2] = 0.

    rot[1, 0] = -sp * st * spsi - cp * cpsi
    rot[1, 1] = sp * st * cpsi - cp * spsi
    rot[1, 2] = 0.

    rot[2, 0] = -cp * st * spsi + sp * cpsi
    rot[2, 1] = cp * st * cpsi + sp * spsi
    rot[2, 2] = 0.

    return rot

def WMat(phi, theta, psi):
    wmat = np.zeros((3, 3))

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    tt = np.tan(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    wmat[0, 0] = 1.
    wmat[0, 1] = sp * tt
    wmat[0, 2] = cp * tt

    wmat[1, 0] = 0.
    wmat[1, 1] = cp
    wmat[1, 2] = -sp

    wmat[2, 0] = 0.
    wmat[2, 1] = sp / ct
    wmat[2, 2] = cp / ct

    return wmat


def dynamics(x_and_u):
    x = x_and_u[0:10, 0]
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]

    u = x_and_u[10:, 0]
    az = u[0]
    p = u[1]
    q = u[2]
    r = u[3]

    R_I_b = RotInertial2Body(phi, theta, psi)
    wmat = WMat(phi, theta, psi)

    grav_I = np.array([0., 0., GRAV])
    grav_b = np.matmul(R_I_b, grav_I)

    vel = x[6:9]
    pqr = u[1:4]

    xdot = np.zeros_like(x)

    xdot[0:3] = np.matmul(R_I_b.transpose(), vel)
    xdot[3:6] = np.matmul(wmat, pqr)

    xdot[6] = grav_b[0] + vel_v * r - vel_w * q - mu * vel_u
    xdot[7] = grav_b[1] + vel_w * p - vel_u * r - mu * vel_v
    xdot[8] = grav_b[2] + vel_u * q - vel_v * p - az

    xdot[9] = 0


    return xdot

def analytical_state_jac(x_and_u):
    x = x_and_u[0:10, 0]
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]

    u = x_and_u[10:]
    az = u[0]
    p = u[1]
    q = u[2]
    r = u[3]

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    tt = np.tan(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    R_I_b = RotInertial2Body(phi, theta, psi)
    vel_b = x[6:9]

    jac = np.zeros((10, 10))

    # pdot
    # jac[0, 3] = (cp * st * cpsi + sp * spsi) * vel_v + (-sp * st * cpsi + cp *
            # spsi) * vel_w
    # jac[0, 4] = (-st * cpsi) * vel_u + (sp * ct * cpsi) * vel_v + (cp * ct *
            # cpsi) * vel_w
    # jac[0, 5] = (-ct * spsi) * vel_u + (-sp * st * spsi - cp * cpsi) * vel_v + \
            # (-cp * st * spsi + sp * cpsi) * vel_w

    # jac[1, 3] = (cp * st * spsi - sp * cpsi) * vel_v + (-sp * st * spsi - cp *
            # cpsi) * vel_w
    # jac[1, 4] = (-st * spsi) * vel_u + (sp * ct * spsi) * vel_v + (cp * ct *
            # spsi) * vel_w
    # jac[1, 5] = (ct * cpsi) * vel_u + (sp * st * cpsi - cp * spsi) * vel_v + (cp * st *
            # cpsi + sp * spsi) * vel_w

    # jac[2, 3] = (cp * ct) * vel_v + (-sp * ct) * vel_w
    # jac[2, 4] = (-ct) * vel_u + (-sp * st) * vel_v + (-cp * st) * vel_w
    # jac[2, 5] = 0.

    jac[0:3, 3] = np.squeeze(np.matmul(RotI2BdPhi(phi, theta, psi).transpose(), vel_b))
    jac[0:3, 4] = np.squeeze(np.matmul(RotI2BdTheta(phi, theta, psi).transpose(), vel_b))
    jac[0:3, 5] = np.squeeze(np.matmul(RotI2BdPsi(phi, theta, psi).transpose(), vel_b))

    jac[0:3, 6:9] = R_I_b.transpose()

    # att dot
    jac[3, 3] = cp * tt * q - sp * tt *r
    jac[3, 4] = (sp / (ct * ct)) * q + (cp / (ct * ct)) * r
    jac[3, 5] = 0.

    jac[4, 3] = -sp * q - cp * r
    jac[4, 4] = 0.
    jac[4, 5] = 0.

    jac[5, 3] = (cp / ct) * q + (-sp / ct) * r
    jac[5, 4] = (sp / ct) * tt * q + (cp  / ct) * tt * r
    jac[5, 5] = 0.

    # vel dot
    jac[6, 3] = 0.
    jac[6, 4] = -ct * GRAV
    jac[6, 5] = 0.
    jac[6, 6] = -mu
    jac[6, 7] = r
    jac[6, 8] = -q
    jac[6, 9] = -vel_u

    jac[7, 3] = cp * ct * GRAV
    jac[7, 4] = -sp * st * GRAV
    jac[7, 5] = 0.
    jac[7, 6] = -r
    jac[7, 7] = -mu
    jac[7, 8] = p
    jac[7, 9] = -vel_v

    jac[8, 3] = -sp * ct * GRAV
    jac[8, 4] = -cp * st * GRAV
    jac[8, 5] = 0.
    jac[8, 6] = q
    jac[8, 7] = -p
    jac[8, 8] = 0.
    jac[8, 9] = 0.

    return jac

def analytical_input_jac(x_and_u):
    x = x_and_u[0:10, 0]
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]

    u = x_and_u[10:]
    az = u[0]
    p = u[1]
    q = u[2]
    r = u[3]

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    tt = np.tan(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    R_I_b = RotInertial2Body(phi, theta, psi)
    wmat = WMat(phi, theta, psi)
    vel = x[6:9]

    jac = np.zeros((10, 4))

    # attitude dot
    jac[3:6, 1:4] = wmat

    # vel dot
    jac[6, 0] = 0.
    jac[6, 1] = 0.
    jac[6, 2] = -vel_w
    jac[6, 3] = vel_v

    jac[7, 0] = 0.
    jac[7, 1] = vel_w
    jac[7, 2] = 0.
    jac[7, 3] = -vel_u

    jac[8, 0] = -1.
    jac[8, 1] = -vel_v
    jac[8, 2] = vel_u
    jac[8, 3] = 0.
    
    return jac

def gps_meas_model(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    tt = np.tan(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    R_I_b = RotInertial2Body(phi, theta, psi)
    pos_I = x[0:3]
    vel_b = x[6:9]

    z = np.zeros((6, 1))
    z[0:3] = pos_I
    z[3:6] = np.matmul(R_I_b.transpose(), vel_b)

    return z

def analytical_gps_meas_jac(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]

    sp = np.sin(phi)
    cp = np.cos(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    tt = np.tan(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    R_I_b = RotInertial2Body(phi, theta, psi)
    pos_I = x[0:3]
    vel_b = x[6:9]

    jac = np.zeros((6, 10))
    # pos
    jac[0:3, 0:3] = np.eye(3)

    # vel
    jac[3:6, 3] = np.squeeze(np.matmul(RotI2BdPhi(phi, theta, psi).transpose(), vel_b))
    jac[3:6, 4] = np.squeeze(np.matmul(RotI2BdTheta(phi, theta, psi).transpose(), vel_b))
    jac[3:6, 5] = np.squeeze(np.matmul(RotI2BdPsi(phi, theta, psi).transpose(), vel_b))

    jac[3:6, 6:9] = R_I_b.transpose()

    return jac

def test_state_jacobian():
    x = np.random.rand(10, 1)
    u = np.random.rand(4, 1)
    x_and_u = np.vstack((x, u))

    jac_func = nd.Jacobian(dynamics)

    xdot = dynamics(x_and_u)
    jac = jac_func(x_and_u)

    state_jac = jac[:, 0:10]
    analytical_jac = analytical_state_jac(x_and_u)

    res = np.isclose(state_jac, analytical_jac)
    assert(np.all(res))

def test_input_jacobian():
    x = np.random.rand(10, 1)
    u = np.random.rand(4, 1)
    x_and_u = np.vstack((x, u))

    jac_func = nd.Jacobian(dynamics)

    xdot = dynamics(x_and_u)
    jac = jac_func(x_and_u)

    input_jac = jac[:, 10:14]
    analytical_jac = analytical_input_jac(x_and_u)

    res = np.isclose(input_jac, analytical_jac)
    assert(np.all(res)), res

def test_gps_model_jacobian():
    x = np.random.rand(10, 1)
    jac_func = nd.Jacobian(gps_meas_model)

    jac = jac_func(x)
    analytical_jac = analytical_gps_meas_jac(x)

    res = np.isclose(jac, analytical_jac)
    assert(np.all(res)), res


if __name__ == '__main__':
    test_state_jacobian()
    test_input_jacobian()
    test_gps_model_jacobian()
