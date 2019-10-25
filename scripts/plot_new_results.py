import numpy as np
from math import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from plotWindow import plotWindow
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#  NUM_LANDMARKS = 20
#  EST_STATE = 16 + 7 + NUM_LANDMARKS * 3
TRUTH = 3 + 4 + 3 + 3 + 3 + 3
EST = 3 + 4 + 3 + 3 + 3 + 3 + 3 + 2 + 1 + 1
COV = 54
LOG_WIDTH = 1 + TRUTH + EST + COV

data = np.reshape(np.fromfile("/tmp/landing_sim.bin", dtype=np.float64), (-1, LOG_WIDTH)).T

# check for NAns
if (np.any(np.isnan(data))):
    print("Uh Oh.... We've got NaNs")

print('data')
print(data.shape)

t = data[0,:]
x = data[1 : 1 + TRUTH, : ]
# x_goal = data[1 + UAV_STATE : 1 + UAV_STATE + LANDING_VEH_STATES, : ]
# x_lms = data[1 + UAV_STATE + LANDING_VEH_STATES : 1 + UAV_STATE +
        # LANDING_VEH_STATES + LANDING_VEH_LMS, : ]
xhat = data[1 + TRUTH : 1 + TRUTH + EST, :]
        # + LANDING_VEH_STATES + LANDING_VEH_LMS + EST_STATE, :]
# phat = data[1 + UAV_STATE + LANDING_VEH_STATES + LANDING_VEH_LMS + EST_STATE : 1
        # + UAV_STATE + LANDING_VEH_STATES + LANDING_VEH_LMS + 2 * EST_STATE, :]
phat = data[1 + TRUTH + EST: 1 + TRUTH + EST + COV, :]

print('x')
print(x.shape)
# print('x_goal')
# print(x_goal.shape)
# print('x_lms')
# print(x_lms.shape)
print('xhat')
print(xhat.shape)
print('phat')
print(phat.shape)

# print(x_lms[:, 0])
# lms = x_lms[:, 0]
# print(np.reshape(lms, (3, 5)).transpose())

pw = plotWindow()

f = plt.figure(dpi=150)
plt.plot()
plt.plot(x[1,:], x[0,:], label="UAV $x$")
plt.plot(xhat[1,:], xhat[0,:], label=r"UAV $\hat{x}$")
plt.legend()
pw.addPlot("2D Position", f)

ylabel = [r'$p_n$', r'$p_e$', r'$p_d$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Position")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i,:], label="x")
    plt.plot(t, xhat[i,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i,:] + 2. * np.sqrt(phat[i, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i,:] - 2. * np.sqrt(phat[i, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Position", f)

ylabel = [r'$\phi$', r'$\theta$', r'$\psi$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Attitude")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i+7,:], label="x")
    plt.plot(t, xhat[i+7,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i+7,:] + 2. * np.sqrt(phat[i+3, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i+7,:] - 2. * np.sqrt(phat[i+3, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Attitude", f)

ylabel = [r'$u$', r'$v$', r'$w$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Velocity")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i+10,:], label="x")
    plt.plot(t, xhat[i+10,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i+10,:] + 2. * np.sqrt(phat[i+6, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i+10,:] - 2. * np.sqrt(phat[i+6, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Velocity", f)

ylabel = [r'$\beta_{ax}$', r'$\beta_{ay}$', r'$\beta_{az}$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Accel Bias")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i+13,:], label="x")
    plt.plot(t, xhat[i+13,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i+13,:] + 2. * np.sqrt(phat[i+9, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i+13,:] - 2. * np.sqrt(phat[i+9, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Accel Bias", f)

ylabel = [r'$\omega_{ax}$', r'$\omega_{ay}$', r'$\omega_{az}$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Gyro Bias")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i+16,:], label="x")
    plt.plot(t, xhat[i+16,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i+16,:] + 2. * np.sqrt(phat[i+12, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i+16,:] - 2. * np.sqrt(phat[i+12, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Gyro Bias", f)

pw.show()


