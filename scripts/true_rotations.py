import numpy as np
from math import *

phi = 1.45
theta = -0.32
psi = 0.567

# define rotation matrix (right handed)
R_roll = np.array([
      [1, 0, 0],
      [0, cos(phi), sin(phi)],
      [0, -sin(phi), cos(phi)]
      ])
R_pitch = np.array([
      [cos(theta), 0, -sin(theta)],
      [0, 1, 0],
      [sin(theta), 0, cos(theta)]
      ])
R_yaw = np.array([
      [cos(psi), sin(psi), 0],
      [-sin(psi), cos(psi), 0],
      [0, 0, 1]
      ])

# Rotation from Inertial frame to body frame
R_I_b = np.matmul(R_roll, np.matmul(R_pitch, R_yaw));  

print("R_I_b")
print(R_I_b)

vec_I = np.array([1., 1., 1])

print("vec_b")
print(R_I_b.dot(vec_I))
