import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from plotWindow import plotWindow
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

UAV_STATE = 13
COMMAND = 4
VEH_STATE = 6
VEH_LMS = 15
EST_STATE = 19
LOG_WIDTH = 1 + UAV_STATE + COMMAND + UAV_STATE + COMMAND + VEH_STATE \
        + VEH_LMS + EST_STATE + EST_STATE

data = np.reshape(np.fromfile("/tmp/landing_sim.bin", dtype=np.float64), (-1, LOG_WIDTH)).T

# check for NAns
if (np.any(np.isnan(data))):
    print("Uh Oh.... We've got NaNs")

t = data[0,:]
x = data[1:14,:]
u = data[14:18,:]
xc = data[18:31, :]
ur = data[31:35, :]

x_veh = data[35:41, :]
x_veh_lms = np.reshape(data[41:56, :], (3, 5, -1))
xhat_veh = data[56:75, :]
phat_veh = data[75:94, :]

pw = plotWindow()

xlabel = [r'$p_x$', r'$p_y$', r'$p_z$',
          r'$q_w$', r'$q_x$', r'$q_y$', r'$q_z$',
          r'$v_x$', r'$v_y$', r'$v_z$',
          r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']

ulabel = [r'$F$',
          r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']

f = plt.figure(dpi=150)
plt.plot()
plt.plot(x[1,:], x[0,:], label="x")
plt.plot(xc[1,:], xc[0,:], label=r"$x_c$")
pw.addPlot("2D Position", f)

f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Position")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i,:], label="x")
    plt.plot(t, xc[i,:], label=r"$x_c$")
    plt.ylabel(xlabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Position", f)

f = plt.figure(dpi=150)
plt.plot()
for i in range(4):
    plt.suptitle("Attitude")
    plt.subplot(4, 1, i+1)
    plt.plot(t, x[i+3,:], label="x")
    plt.plot(t, xc[i+3,:], label=r"$x_c$")
    plt.ylabel(xlabel[i+3])
    if i == 0:
        plt.legend()
pw.addPlot("Attitude", f)

f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Velocity")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i+7,:], label="x")
    plt.plot(t, xc[i+7,:], label=r"$x_c$")
    plt.ylabel(xlabel[i+7])
    if i == 0:
        plt.legend()
pw.addPlot("Velocity", f)

f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Omega")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i+10,:], label="x")
    plt.plot(t, xc[i+10,:], label=r"$x_c$")
    # plt.ylabel(xlabel[i+10])
    if i == 0:
        plt.legend()
pw.addPlot("Omega", f)

f = plt.figure(dpi=150)
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(t, u[i, :], label='u')
    plt.ylabel(ulabel[i])
pw.addPlot("Input", f)

# vehicle Estimates
# x_veh = data[35:41, :]
# x_veh_lms = np.reshape(data[41:56, :], (3, 5, -1))
# xhat_veh = data[56:62, :]
# phat_veh = data[62:68, :]

# Create velocities in inertial frame
num_steps = len(t)
veh_v_I = np.zeros((2, num_steps));
est_v_I = np.zeros((2, num_steps));

true_r1_I = np.zeros((2, num_steps));
true_r2_I = np.zeros((2, num_steps));
true_r3_I = np.zeros((2, num_steps));
true_r4_I = np.zeros((2, num_steps));
est_r1_I = np.zeros((2, num_steps));
est_r2_I = np.zeros((2, num_steps));
est_r3_I = np.zeros((2, num_steps));
est_r4_I = np.zeros((2, num_steps));

for i in range(num_steps):
    true_theta = x_veh[4, i]
    true_R_I_b = np.array([[np.cos(true_theta), np.sin(true_theta)],
                 [-np.sin(true_theta), np.cos(true_theta)]])
    true_veh_I = np.matmul(true_R_I_b.transpose(), np.array([x_veh[2, i],
        x_veh[3, i]]))
    veh_v_I[0, i] = true_veh_I[0]
    veh_v_I[1, i] = true_veh_I[1]

    true_r1_I[:, i] = np.matmul(true_R_I_b.transpose(),
            np.array([x_veh_lms[0, 1, i], x_veh_lms[1, 1, i]]))
    true_r2_I[:, i] = np.matmul(true_R_I_b.transpose(),
            np.array([x_veh_lms[0, 2, i], x_veh_lms[1, 2, i]]))
    true_r3_I[:, i] = np.matmul(true_R_I_b.transpose(),
            np.array([x_veh_lms[0, 3, i], x_veh_lms[1, 3, i]]))
    true_r4_I[:, i] = np.matmul(true_R_I_b.transpose(),
            np.array([x_veh_lms[0, 4, i], x_veh_lms[1, 4, i]]))

    est_theta = xhat_veh[5, i]
    est_R_I_b = np.array([[np.cos(est_theta), np.sin(est_theta)],
                 [-np.sin(est_theta), np.cos(est_theta)]])
    est_veh_I = np.matmul(est_R_I_b.transpose(), np.array([xhat_veh[3, i],
            xhat_veh[4, i]]))
    est_v_I[0, i] = est_veh_I[0]
    est_v_I[1, i] = est_veh_I[1]

    est_r1_I[:, i] = np.matmul(est_R_I_b.transpose(),
            np.array([xhat_veh[7, i], xhat_veh[8, i]]))
    est_r2_I[:, i] = np.matmul(est_R_I_b.transpose(),
            np.array([xhat_veh[10, i], xhat_veh[11, i]]))
    est_r3_I[:, i] = np.matmul(est_R_I_b.transpose(),
            np.array([xhat_veh[13, i], xhat_veh[14, i]]))
    est_r4_I[:, i] = np.matmul(est_R_I_b.transpose(),
            np.array([xhat_veh[16, i], xhat_veh[17, i]]))

f = plt.figure(dpi=150)
plt.plot()
ylabel = [r"$p_x$", r"$p_y$", r"$\rho$"]
for i in range(3):
    plt.suptitle("Vehicle Pos UAV Veh Frame")
    plt.subplot(3, 1, i+1)
    if i == 2:
        true_rho = -1. / x[2, :]
        plt.plot(t, true_rho, label="x")
    else:
        plt.plot(t, x_veh[i,:], label="x")
    plt.plot(t, xhat_veh[i,:], '--', label=r"$\hat{x}$")
    pos_cov = xhat_veh[i,:] + 2. * np.sqrt(phat_veh[i, :])
    neg_cov = xhat_veh[i,:] - 2. * np.sqrt(phat_veh[i, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Veh Pos", f)

f = plt.figure(dpi=150)
plt.plot()
ylabel = [r"$v_x$", r"$v_y$"]
for i in range(2):
    plt.suptitle("Vehicle Vel, Veh Body Frame")
    plt.subplot(2, 1, i+1)
    plt.plot(t, x_veh[2 + i,:], label="x")
    plt.plot(t, xhat_veh[3 + i,:], '--', label=r"$\hat{x}$")
    pos_cov = xhat_veh[3 + i,:] + 2. * np.sqrt(phat_veh[3 + i, :])
    neg_cov = xhat_veh[3 + i,:] - 2. * np.sqrt(phat_veh[3 + i, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Veh Vel Body", f)

f = plt.figure(dpi=150)
plt.plot()
ylabel = [r"$v_x$", r"$v_y$"]
for i in range(2):
    plt.suptitle("Vehicle Vel, Veh Inertial Frame")
    plt.subplot(2, 1, i+1)
    plt.plot(t, veh_v_I[i,:], label="x")
    plt.plot(t, est_v_I[i,:], '--', label=r"$\hat{x}$")
    pos_cov = est_v_I[i,:] + 2. * np.sqrt(phat_veh[3 + i, :])
    neg_cov = est_v_I[i,:] - 2. * np.sqrt(phat_veh[3 + i, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Veh Vel Inertial", f)

f = plt.figure(dpi=150)
plt.plot()
ylabel = [r"$\theta_I^b$", r"$\omega$"]
for i in range(2):
    plt.suptitle("Vehicle Attitude")
    plt.subplot(2, 1, i+1)
    plt.plot(t, x_veh[4 + i,:], label="x")
    plt.plot(t, xhat_veh[5 + i,:], '--', label=r"$\hat{x}$")
    pos_cov = xhat_veh[5 + i,:] + 2. * np.sqrt(phat_veh[5 + i, :])
    neg_cov = xhat_veh[5 + i,:] - 2. * np.sqrt(phat_veh[5 + i, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Veh Att", f)

# x_veh = data[35:41, :]
# x_veh_lms = np.reshape(data[41:56, :], (3, 5, -1))
# xhat_veh = data[56:75, :]
# phat_veh = data[75:94, :]

f = plt.figure(dpi=150)
plt.plot()
ylabel = [r"$LM 1$", r"$LM 2$", r"$LM 3$"]
for i in range(3):
    plt.suptitle("Landmark Points, Veh Body Frame")
    plt.subplot(3, 3, 3*i+1)
    lm_idx = i + 1
    plt.plot(t, x_veh_lms[0, lm_idx, :], label=r"$r_x$")
    rx_idx = 7 + 3 * i
    plt.plot(t, xhat_veh[rx_idx, :], '--', label=r"$\hat{r}_x$")
    pos_cov = xhat_veh[rx_idx, :] + 2. * np.sqrt(phat_veh[rx_idx, :])
    neg_cov = xhat_veh[rx_idx, :] - 2. * np.sqrt(phat_veh[rx_idx, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()

    plt.subplot(3, 3, 3*i+2)
    plt.plot(t, x_veh_lms[1, lm_idx, :], label=r"$r_y$")
    ry_idx = 8 + 3 * i
    plt.plot(t, xhat_veh[ry_idx, :], '--', label=r"$\hat{r}_y$")
    pos_cov = xhat_veh[ry_idx, :] + 2. * np.sqrt(phat_veh[ry_idx, :])
    neg_cov = xhat_veh[ry_idx, :] - 2. * np.sqrt(phat_veh[ry_idx, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    if i == 0:
        plt.legend()

    plt.subplot(3, 3, 3*i+3)
    true_rho = -1. / (x[2, :] + x_veh_lms[2, lm_idx, :])
    plt.plot(t, true_rho, label=r"$\rho$")
    rhohat_idx = 9 + 3 * i
    plt.plot(t, xhat_veh[rhohat_idx,:], '--', label=r"$\hat{\rho}$")
    pos_cov = xhat_veh[rhohat_idx,:] + 2. * np.sqrt(phat_veh[rhohat_idx, :])
    neg_cov = xhat_veh[rhohat_idx,:] - 2. * np.sqrt(phat_veh[rhohat_idx, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    if i == 0:
        plt.legend()
pw.addPlot("Landmarks Body", f)

f = plt.figure(dpi=150)
plt.plot()
true_r = [true_r1_I, true_r2_I, true_r3_I, true_r4_I]
est_r = [est_r1_I, est_r2_I, est_r3_I, est_r4_I]
ylabel = [r"$LM 1$", r"$LM 2$", r"$LM 3$"]
for i in range(3):
    plt.suptitle("Landmark Points, Veh Inertial Frame")
    plt.subplot(3, 3, 3*i+1)
    lm_idx = i + 1
    plt.plot(t, true_r[i][0, :], label=r"$r_x$")
    rx_idx = 7 + 3 * i
    plt.plot(t, est_r[i][0, :], '--', label=r"$\hat{r}_x$")
    pos_cov = est_r[i][0, :] + 2. * np.sqrt(phat_veh[rx_idx, :])
    neg_cov = est_r[i][0, :] - 2. * np.sqrt(phat_veh[rx_idx, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()

    plt.subplot(3, 3, 3*i+2)
    plt.plot(t, true_r[i][1, :], label=r"$r_y$")
    ry_idx = 8 + 3 * i
    plt.plot(t, est_r[i][1, :], '--', label=r"$\hat{r}_y$")
    pos_cov = est_r[i][1, :] + 2. * np.sqrt(phat_veh[ry_idx, :])
    neg_cov = est_r[i][1, :] - 2. * np.sqrt(phat_veh[ry_idx, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    if i == 0:
        plt.legend()

    plt.subplot(3, 3, 3*i+3)
    true_rho = -1. / (x[2, :] + x_veh_lms[2, lm_idx, :])
    plt.plot(t, true_rho, label=r"$\rho$")
    rhohat_idx = 9 + 3 * i
    plt.plot(t, xhat_veh[rhohat_idx,:], '--', label=r"$\hat{\rho}$")
    pos_cov = xhat_veh[rhohat_idx,:] + 2. * np.sqrt(phat_veh[rhohat_idx, :])
    neg_cov = xhat_veh[rhohat_idx,:] - 2. * np.sqrt(phat_veh[rhohat_idx, :])
    plt.plot(t, pos_cov, 'r--', label=r"$2\sigma bound$")
    plt.plot(t, neg_cov, 'r--', label=r"$2\sigma bound$")
    if i == 0:
        plt.legend()
pw.addPlot("Landmarks Inertial", f)

pw.show()


