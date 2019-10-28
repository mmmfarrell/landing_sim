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

TRUTH = 3 + 4 + 3 + 3 + 3 + 3 + 3 + 2 + 1 + 1 + 30
EST = 3 + 4 + 3 + 3 + 3 + 3 + 3 + 2 + 1 + 1 + 30
COV = 54
LOG_WIDTH = 1 + TRUTH + EST + COV

run = 1
filename = "/home/mmmfarrell/magicc/landing_sim/mc_results/with_landmarks_" + str(run) + ".bin"
data = np.reshape(np.fromfile(filename, dtype=np.float64), (-1, LOG_WIDTH)).T

# check for NAns
if (np.any(np.isnan(data))):
    print("Uh Oh.... We've got NaNs")

print('data')
print(data.shape)

t = data[0,:]
x = data[1 : 1 + TRUTH, : ]
x_lms = np.reshape(x[26:, :], (10, 3, -1))
xhat = data[1 + TRUTH : 1 + TRUTH + EST, :]
xhat_lms = np.reshape(xhat[26:, :], (10, 3, -1))
phat = data[1 + TRUTH + EST : 1 + TRUTH + EST + COV, :]

print('x')
print(x.shape)
print('xlms')
print(x_lms.shape)
print('xhat')
print(xhat.shape)
print('xhat_lms')
print(xhat_lms.shape)
print('phat')
print(phat.shape)

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

ylabel = [r'$p_x$', r'$p_y$', r'$p_z$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(3):
    plt.suptitle("Goal Position")
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[i+19,:], label="x")
    plt.plot(t, xhat[i+19,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i+19,:] + 2. * np.sqrt(phat[i+17, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i+19,:] - 2. * np.sqrt(phat[i+17, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Goal Position", f)

ylabel = [r'$v_x$', r'$v_y$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(2):
    plt.suptitle("Goal Velocity")
    plt.subplot(2, 1, i+1)
    plt.plot(t, x[i+22,:], label="x")
    plt.plot(t, xhat[i+22,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i+22,:] + 2. * np.sqrt(phat[i+20, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i+22,:] - 2. * np.sqrt(phat[i+20, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Goal Velocity", f)

ylabel = [r'$\theta$', r'$\omega$']
f = plt.figure(dpi=150)
plt.plot()
for i in range(2):
    plt.suptitle("Goal Attitude")
    plt.subplot(2, 1, i+1)
    plt.plot(t, x[i+24,:], label="x")
    plt.plot(t, xhat[i+24,:], label=r"$\hat{x}$")
    plt.plot(t, xhat[i+24,:] + 2. * np.sqrt(phat[i+22, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.plot(t, xhat[i+24,:] - 2. * np.sqrt(phat[i+22, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Goal Attitude", f)

ylabel = [r'$x$', r'$y$', r'$z$']
f = plt.figure(dpi=150)
plt.plot()
plt.suptitle("Landmarks")
for lm in range(4):
    for i in range(3):
        plt.subplot(3, 4, lm+i*4+1)
        plt.plot(t, x_lms[lm, i, :], label="x")
        plt.plot(t, xhat_lms[lm, i, :], label=r"$\hat{x}$")
        lm_idx = 24 + 3 * lm
        plt.plot(t, xhat_lms[lm, i, :] + 2. * np.sqrt(phat[i+lm_idx, :]), '-k', alpha=0.3, label=r"$2\sigma$")
        plt.plot(t, xhat_lms[lm, i, :] - 2. * np.sqrt(phat[i+lm_idx, :]), '-k', alpha=0.3, label=r"$2\sigma$")
    plt.ylabel(ylabel[i])
    if i == 0:
        plt.legend()
pw.addPlot("Landmarks", f)

pw.show()


