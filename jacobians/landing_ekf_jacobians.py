import pytest
import numpy as np
import numdifftools as nd

GRAV = 9.81
STATES = 10 + 10
INPUTS = 4

RBC = np.array([[0., 1., 0.],
                [-1., 0., 0],
                [0., 0., 1]])
# PCBB = np.reshape(np.array([1., -2., 0.]), (3, 1))
PCBB = np.reshape(np.array([0., 0., 0.]), (3, 1))
E1 = np.array([1., 0., 0])
E2 = np.array([0., 1., 0])
E3 = np.array([0., 0., 1])
FX = 410.
FY = 420.
CX = 320.
CY = 240.

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

def Rot2DInertial2Body(theta):
    rot = np.zeros((2, 2))

    st = np.sin(theta)
    ct = np.cos(theta)

    rot[0, 0] = ct
    rot[0, 1] = st

    rot[1, 0] = -st
    rot[1, 1] = ct

    return rot

def Rot2DdTheta(theta):
    rot = np.zeros((2, 2))

    st = np.sin(theta)
    ct = np.cos(theta)

    rot[0, 0] = -st
    rot[0, 1] = ct

    rot[1, 0] = -ct
    rot[1, 1] = -st

    return rot

def Rot3DInertial2Body(theta):
    rot = np.zeros((3, 3))

    st = np.sin(theta)
    ct = np.cos(theta)

    rot[0, 0] = ct
    rot[0, 1] = st

    rot[1, 0] = -st
    rot[1, 1] = ct

    rot[2, 2] = 1.

    return rot

def Rot3DdTheta(theta):
    rot = np.zeros((3, 3))

    st = np.sin(theta)
    ct = np.cos(theta)

    rot[0, 0] = -st
    rot[0, 1] = ct

    rot[1, 0] = -ct
    rot[1, 1] = -st

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
    x = x_and_u[0:STATES, 0]
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    # px_g = x[10]
    # py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    u = x_and_u[STATES:, 0]
    az = u[0]
    p = u[1]
    q = u[2]
    r = u[3]

    R_I_b = RotInertial2Body(phi, theta, psi)
    wmat = WMat(phi, theta, psi)

    grav_I = np.array([0., 0., GRAV])
    grav_b = np.matmul(R_I_b, grav_I)

    vel_b = x[6:9]
    pqr = u[1:4]

    xdot = np.zeros_like(x)

    # UAV dynamics
    xdot[0:3] = np.matmul(R_I_b.transpose(), vel_b)
    xdot[3:6] = np.matmul(wmat, pqr)

    xdot[6] = grav_b[0] + vel_v * r - vel_w * q - mu * vel_u
    xdot[7] = grav_b[1] + vel_w * p - vel_u * r - mu * vel_v
    xdot[8] = grav_b[2] + vel_u * q - vel_v * p - az

    xdot[9] = 0

    # Landing target dynamics
    R_v_g = Rot2DInertial2Body(theta_g)
    vel_g = x[13:15]
    I_2x3 = np.eye(3)[0:2, :]
    e3 = np.eye(3)[2, :]

    xdot[10:12] = np.matmul(R_v_g.transpose(), vel_g) - np.matmul(I_2x3,
        np.matmul(R_I_b.transpose(), vel_b))
    xdot[12] = rho_g * rho_g * np.matmul(e3.transpose(),
            np.matmul(R_I_b.transpose(), vel_b))
    # xdot[13:15] = 0
    xdot[15] = omega_g
    # xdot[16] = 0
    # xdot[17:20] = 0

    return xdot

def analytical_state_jac(x_and_u):
    x = x_and_u[0:STATES, 0]
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    # px_g = x[10]
    # py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    u = x_and_u[STATES:]
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

    jac = np.zeros((STATES, STATES))

    # dUAV / dUAV
    # pdot
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

    # dUAV / dGOAL = 0

    # Landing target dynamics
    R_v_g = Rot2DInertial2Body(theta_g)
    dRdTheta = Rot2DdTheta(theta_g)
    vel_g = x[13:15]
    I_2x3 = np.eye(3)[0:2, :]
    e3 = np.eye(3)[2, :]

    # dGOAL / dUAV
    jac[10:12, 3] = np.squeeze(np.matmul(np.matmul(-I_2x3, RotI2BdPhi(phi,
        theta, psi).transpose()), vel_b))
    jac[10:12, 4] = np.squeeze(np.matmul(np.matmul(-I_2x3, RotI2BdTheta(phi,
        theta, psi).transpose()), vel_b))
    jac[10:12, 5] = np.squeeze(np.matmul(np.matmul(-I_2x3, RotI2BdPsi(phi,
        theta, psi).transpose()), vel_b))

    jac[10:12, 6:9] = np.matmul(-I_2x3, R_I_b.transpose())

    jac[12, 3] = rho_g * rho_g * np.squeeze(np.matmul(np.matmul(e3.transpose(), RotI2BdPhi(phi,
        theta, psi).transpose()), vel_b))
    jac[12, 4] = rho_g * rho_g * np.squeeze(np.matmul(np.matmul(e3.transpose(), RotI2BdTheta(phi,
        theta, psi).transpose()), vel_b))
    jac[12, 5] = rho_g * rho_g * np.squeeze(np.matmul(np.matmul(e3.transpose(), RotI2BdPsi(phi,
        theta, psi).transpose()), vel_b))

    jac[12, 6:9] = rho_g * rho_g * np.matmul(e3.transpose(), R_I_b.transpose())

    # dGOAL / dGOAL
    jac[10:12, 13:15] = R_v_g.transpose()
    jac[10:12, 15] = np.matmul(dRdTheta.transpose(), vel_g)

    jac[12, 12] = 2. * rho_g * np.matmul(e3.transpose(),
            np.matmul(R_I_b.transpose(), vel_b))
    jac[15, 16] = 1.

    return jac

def analytical_input_jac(x_and_u):
    x = x_and_u[0:STATES, 0]
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]

    u = x_and_u[STATES:]
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

    jac = np.zeros((STATES, INPUTS))

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

    jac = np.zeros((6, STATES))
    # pos
    jac[0:3, 0:3] = np.eye(3)

    # vel
    jac[3:6, 3] = np.squeeze(np.matmul(RotI2BdPhi(phi, theta, psi).transpose(), vel_b))
    jac[3:6, 4] = np.squeeze(np.matmul(RotI2BdTheta(phi, theta, psi).transpose(), vel_b))
    jac[3:6, 5] = np.squeeze(np.matmul(RotI2BdPsi(phi, theta, psi).transpose(), vel_b))

    jac[3:6, 6:9] = R_I_b.transpose()

    return jac

def goal_pix_meas_model(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    px_g = x[10]
    py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    R_I_b = RotInertial2Body(phi, theta, psi)

    p_g_v_v = np.array([px_g, py_g, 1. / rho_g])

    p_g_c_c = np.matmul(RBC, np.matmul(R_I_b, p_g_v_v) - PCBB)

    pix_x = FX * (p_g_c_c[0] / p_g_c_c[2]) + CX
    pix_y = FY * (p_g_c_c[1] / p_g_c_c[2]) + CY

    return np.array([pix_x, pix_y])

def analytical_goal_pix_jac(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    px_g = x[10]
    py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    R_I_b = RotInertial2Body(phi, theta, psi)

    p_g_v_v = np.array([px_g, py_g, 1. / rho_g])

    p_g_c_c = np.matmul(RBC, np.matmul(R_I_b, p_g_v_v) - PCBB)

    pix_x = FX * (p_g_c_c[0] / p_g_c_c[2]) + CX
    pix_y = FY * (p_g_c_c[1] / p_g_c_c[2]) + CY

    jac = np.zeros((2, STATES))

    # d / d (p_g_v_v)
    d1 = -np.matmul(E3, np.matmul(RBC, R_I_b)) * FX * (p_g_c_c[0] / p_g_c_c[2] /
            p_g_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, R_I_b)) \
                    / p_g_c_c[2]
    d2 = -np.matmul(E3, np.matmul(RBC, R_I_b)) * FY * (p_g_c_c[1] / p_g_c_c[2] /
            p_g_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, R_I_b)) \
                    / p_g_c_c[2]
    jac[0, 10:12] = d1[0:2]
    jac[1, 10:12] = d2[0:2]

    # d / d rho
    jac[0, 12] = -d1[2] / rho_g / rho_g
    jac[1, 12] = -d2[2] / rho_g / rho_g

    # d / d (phi, theta, psi)
    dRdPhi = RotI2BdPhi(phi, theta, psi)
    dRdTheta= RotI2BdTheta(phi, theta, psi)
    dRdPsi= RotI2BdPsi(phi, theta, psi)

    d1dphi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPhi, p_g_v_v))) * FX * (p_g_c_c[0] / p_g_c_c[2] /
            p_g_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(dRdPhi,
                p_g_v_v))) / p_g_c_c[2]
    d1dtheta = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdTheta, p_g_v_v))) * FX * (p_g_c_c[0] / p_g_c_c[2] /
            p_g_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(dRdTheta,
                p_g_v_v))) / p_g_c_c[2]
    d1dpsi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPsi, p_g_v_v))) * FX * (p_g_c_c[0] / p_g_c_c[2] /
            p_g_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(dRdPsi,
                p_g_v_v))) / p_g_c_c[2]

    jac[0, 3] = d1dphi
    jac[0, 4] = d1dtheta
    jac[0, 5] = d1dpsi

    d2dphi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPhi, p_g_v_v))) * FY * (p_g_c_c[1] / p_g_c_c[2] /
            p_g_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(dRdPhi,
                p_g_v_v))) / p_g_c_c[2]
    d2dtheta = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdTheta, p_g_v_v))) * FY * (p_g_c_c[1] / p_g_c_c[2] /
            p_g_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(dRdTheta,
                p_g_v_v))) / p_g_c_c[2]
    d2dpsi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPsi, p_g_v_v))) * FY * (p_g_c_c[1] / p_g_c_c[2] /
            p_g_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(dRdPsi,
                p_g_v_v))) / p_g_c_c[2]

    jac[1, 3] = d2dphi
    jac[1, 4] = d2dtheta
    jac[1, 5] = d2dpsi

    return jac

def goal_depth_meas_model(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    px_g = x[10]
    py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    R_I_b = RotInertial2Body(phi, theta, psi)

    p_g_v_v = np.array([px_g, py_g, 1. / rho_g])

    p_g_c_c = np.matmul(RBC, np.matmul(R_I_b, p_g_v_v) - PCBB)

    z_depth = p_g_c_c[2]

    return np.array([z_depth])

def analytical_goal_depth_jac(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    px_g = x[10]
    py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    R_I_b = RotInertial2Body(phi, theta, psi)

    p_g_v_v = np.array([px_g, py_g, 1. / rho_g])

    p_g_c_c = np.matmul(RBC, np.matmul(R_I_b, p_g_v_v) - PCBB)

    jac = np.zeros((1, STATES))

    dRdPhi = RotI2BdPhi(phi, theta, psi)
    dRdTheta= RotI2BdTheta(phi, theta, psi)
    dRdPsi= RotI2BdPsi(phi, theta, psi)

    jac[0, 3] = np.matmul(E3.transpose(), np.matmul(RBC, np.matmul(dRdPhi,
        p_g_v_v)))
    jac[0, 4] = np.matmul(E3.transpose(), np.matmul(RBC, np.matmul(dRdTheta,
        p_g_v_v)))
    jac[0, 5] = np.matmul(E3.transpose(), np.matmul(RBC, np.matmul(dRdPsi,
        p_g_v_v)))

    dzdp = np.matmul(E3.transpose(), np.matmul(RBC, R_I_b))
    jac[0, 10:12] = dzdp[0:2]
    jac[0, 12] = (-1. / rho_g / rho_g) * dzdp[2]

    return jac

def landmark_pix_meas_model(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    px_g = x[10]
    py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    R_I_b = RotInertial2Body(phi, theta, psi)
    R_v_g = Rot3DInertial2Body(theta_g)

    p_i_g_g = np.array([rx_i, ry_i, rz_i])
    p_i_g_v = np.matmul(R_v_g.transpose(), p_i_g_g)

    p_g_v_v = np.array([px_g, py_g, 1. / rho_g])
    p_i_v_v = p_i_g_v + p_g_v_v

    p_i_c_c = np.matmul(RBC, np.matmul(R_I_b, p_i_v_v) - PCBB)

    pix_x = FX * (p_i_c_c[0] / p_i_c_c[2]) + CX
    pix_y = FY * (p_i_c_c[1] / p_i_c_c[2]) + CY

    return np.array([pix_x, pix_y])

def analytical_landmark_pix_jac(x):
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vel_u = x[6]
    vel_v = x[7]
    vel_w = x[8]
    mu = x[9]
    px_g = x[10]
    py_g = x[11]
    rho_g = x[12]
    vx_g = x[13]
    vy_g = x[14]
    theta_g = x[15]
    omega_g = x[16]
    rx_i = x[17]
    ry_i = x[18]
    rz_i = x[19]

    R_I_b = RotInertial2Body(phi, theta, psi)
    R_v_g = Rot3DInertial2Body(theta_g)

    p_i_g_g = np.array([rx_i, ry_i, rz_i])
    p_i_g_v = np.matmul(R_v_g.transpose(), p_i_g_g)

    p_g_v_v = np.array([px_g, py_g, 1. / rho_g])
    p_i_v_v = p_i_g_v + p_g_v_v

    p_i_c_c = np.matmul(RBC, np.matmul(R_I_b, p_i_v_v) - PCBB)

    pix_x = FX * (p_i_c_c[0] / p_i_c_c[2]) + CX
    pix_y = FY * (p_i_c_c[1] / p_i_c_c[2]) + CY

    jac = np.zeros((2, STATES))

    # d / d (p_g_v_v)
    d1 = -np.matmul(E3, np.matmul(RBC, R_I_b)) * FX * (p_i_c_c[0] / p_i_c_c[2] /
            p_i_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, R_I_b)) \
                    / p_i_c_c[2]
    d2 = -np.matmul(E3, np.matmul(RBC, R_I_b)) * FY * (p_i_c_c[1] / p_i_c_c[2] /
            p_i_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, R_I_b)) \
                    / p_i_c_c[2]
    jac[0, 10:12] = d1[0:2]
    jac[1, 10:12] = d2[0:2]

    # # d / d rho_g
    jac[0, 12] = -d1[2] / rho_g / rho_g
    jac[1, 12] = -d2[2] / rho_g / rho_g

    # d / d (phi, theta, psi)
    dRdPhi = RotI2BdPhi(phi, theta, psi)
    dRdTheta= RotI2BdTheta(phi, theta, psi)
    dRdPsi= RotI2BdPsi(phi, theta, psi)

    d1dphi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPhi, p_i_v_v))) * FX * (p_i_c_c[0] / p_i_c_c[2] /
            p_i_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(dRdPhi,
                p_i_v_v))) / p_i_c_c[2]
    d1dtheta = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdTheta, p_i_v_v))) * FX * (p_i_c_c[0] / p_i_c_c[2] /
            p_i_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(dRdTheta,
                p_i_v_v))) / p_i_c_c[2]
    d1dpsi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPsi, p_i_v_v))) * FX * (p_i_c_c[0] / p_i_c_c[2] /
            p_i_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(dRdPsi,
                p_i_v_v))) / p_i_c_c[2]

    jac[0, 3] = d1dphi
    jac[0, 4] = d1dtheta
    jac[0, 5] = d1dpsi

    d2dphi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPhi, p_i_v_v))) * FY * (p_i_c_c[1] / p_i_c_c[2] /
            p_i_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(dRdPhi,
                p_i_v_v))) / p_i_c_c[2]
    d2dtheta = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdTheta, p_i_v_v))) * FY * (p_i_c_c[1] / p_i_c_c[2] /
            p_i_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(dRdTheta,
                p_i_v_v))) / p_i_c_c[2]
    d2dpsi = -np.matmul(E3, np.matmul(RBC, np.matmul(dRdPsi, p_i_v_v))) * FY * (p_i_c_c[1] / p_i_c_c[2] /
            p_i_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(dRdPsi,
                p_i_v_v))) / p_i_c_c[2]

    jac[1, 3] = d2dphi
    jac[1, 4] = d2dtheta
    jac[1, 5] = d2dpsi

    # d / d theta_g
    d_theta_R_v_g = Rot3DdTheta(theta_g)

    d_theta_p_i_g_v = np.matmul(d_theta_R_v_g.transpose(), p_i_g_g)
    # d_theta_p_i_v_v_2d = d_theta_p_i_g_v
    d_theta_p_i_v_v = d_theta_p_i_g_v

    # d_theta_p_i_v_v = np.array([d_theta_p_i_v_v_2d[0], d_theta_p_i_v_v_2d[1], [0.]])

    d1dtheta_g = -np.matmul(E3, np.matmul(RBC, np.matmul(R_I_b,
        d_theta_p_i_v_v))) * FX * (p_i_c_c[0] / p_i_c_c[2] /
            p_i_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(R_I_b,
                d_theta_p_i_v_v))) / p_i_c_c[2]
    d2dtheta_g = -np.matmul(E3, np.matmul(RBC, np.matmul(R_I_b,
        d_theta_p_i_v_v))) * FY * (p_i_c_c[1] / p_i_c_c[2] /
            p_i_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(R_I_b,
                d_theta_p_i_v_v))) / p_i_c_c[2]

    jac[0, 15] = d1dtheta_g
    jac[1, 15] = d2dtheta_g

    # d / d p_i_g_g
    d_rxy_p_i_v_v = R_v_g.transpose()

    d1drxy = -np.matmul(E3, np.matmul(RBC, np.matmul(R_I_b,
        d_rxy_p_i_v_v))) * FX * (p_i_c_c[0] / p_i_c_c[2] /
            p_i_c_c[2]) + FX * np.matmul(E1, np.matmul(RBC, np.matmul(R_I_b,
                d_rxy_p_i_v_v))) / p_i_c_c[2]
    d2drxy = -np.matmul(E3, np.matmul(RBC, np.matmul(R_I_b,
        d_rxy_p_i_v_v))) * FY * (p_i_c_c[1] / p_i_c_c[2] /
            p_i_c_c[2]) + FY * np.matmul(E2, np.matmul(RBC, np.matmul(R_I_b,
                d_rxy_p_i_v_v))) / p_i_c_c[2]

    jac[0, 17:20] = d1drxy
    jac[1, 17:20] = d2drxy

    return jac

def test_state_jacobian():
    x = np.random.rand(STATES, 1)
    u = np.random.rand(INPUTS, 1)
    x_and_u = np.vstack((x, u))

    jac_func = nd.Jacobian(dynamics)

    xdot = dynamics(x_and_u)
    jac = jac_func(x_and_u)

    state_jac = jac[:, 0:STATES]
    analytical_jac = analytical_state_jac(x_and_u)

    res = np.isclose(state_jac, analytical_jac)
    assert(np.all(res)), res

def test_input_jacobian():
    x = np.random.rand(STATES, 1)
    u = np.random.rand(INPUTS, 1)
    x_and_u = np.vstack((x, u))

    jac_func = nd.Jacobian(dynamics)

    xdot = dynamics(x_and_u)
    jac = jac_func(x_and_u)

    input_jac = jac[:, STATES:]
    analytical_jac = analytical_input_jac(x_and_u)

    res = np.isclose(input_jac, analytical_jac)
    assert(np.all(res)), res

def test_gps_model_jacobian():
    x = np.random.rand(STATES, 1)
    jac_func = nd.Jacobian(gps_meas_model)

    jac = jac_func(x)
    analytical_jac = analytical_gps_meas_jac(x)

    res = np.isclose(jac, analytical_jac)
    assert(np.all(res)), res

def test_goal_pix_model_jacobian():
    x = np.random.rand(STATES, 1)
    # x = np.zeros((STATES, 1))
    # x[3] = 0.1
    # x[4] = -0.3
    # x[5] = 1.2
    # x[10] = -0.45
    # x[11] = 1.23
    # x[12] = 0.1
    jac_func = nd.Jacobian(goal_pix_meas_model)

    meas = goal_pix_meas_model(x)
    # print('x = {}'.format(x))
    # print('meas = {}'.format(meas))
    jac = jac_func(x)
    analytical_jac = analytical_goal_pix_jac(x)
    # print('jac = {}'.format(jac))

    res = np.isclose(jac, analytical_jac)
    assert(np.all(res)), res

def test_goal_depth_model_jacobian():
    x = np.random.rand(STATES, 1)
    # x = np.zeros((STATES, 1))
    # x[3] = 0.1
    # x[4] = -0.3
    # x[5] = 1.2
    # x[10] = -0.45
    # x[11] = 1.23
    # x[12] = 0.1
    jac_func = nd.Jacobian(goal_depth_meas_model)

    meas = goal_depth_meas_model(x)
    # print('x = {}'.format(x))
    # print('meas = {}'.format(meas))
    jac = jac_func(x)
    analytical_jac = analytical_goal_depth_jac(x)
    # print('jac = {}'.format(jac))

    res = np.isclose(jac, analytical_jac)
    assert(np.all(res)), res

def test_landmark_pixel_model_jacobian():
    # x = np.random.rand(STATES, 1)

    x = np.zeros((STATES, 1))
    # Simple State
    # x[12] = 0.1
    # x[17] = 1.
    # x[18] = 0.5
    # x[19] = 1.

    # # Complex State
    x[3] = 0.1
    x[4] = -0.3
    x[5] = 1.2
    x[10] = -0.45
    x[11] = 1.23
    x[12] = 0.1
    x[15] = np.pi / 4. # theta_g
    x[17] = 2.34
    x[18] = -0.75
    x[19] = 1.134
    jac_func = nd.Jacobian(landmark_pix_meas_model)

    meas = landmark_pix_meas_model(x)
    print('x = {}'.format(x))
    print('meas = {}'.format(meas))
    jac = jac_func(x)
    analytical_jac = analytical_landmark_pix_jac(x)
    print('jac = {}'.format(jac))

    res = np.isclose(jac, analytical_jac)
    assert(np.all(res)), res


if __name__ == '__main__':
    test_state_jacobian()
    test_input_jacobian()
    test_gps_model_jacobian()
    test_goal_pix_model_jacobian()
    test_goal_depth_model_jacobian()
    test_landmark_pixel_model_jacobian()
