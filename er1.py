import matplotlib.pyplot as plt
import math

import numpy as np

# length of limps
l1 = 0.3
l2 = 0.3
l3 = 0.05

# angle of limps
theta1 = -57
theta2 = -66
theta3 = -57


# forward kinematics transformation matrixes
T01 = np.array([[np.cos(math.radians(theta1)), 0, np.sin(math.radians(theta1)), 0],
                [0, 1, 0, 0],
                [-np.sin(math.radians(theta1)), 0, np.cos(math.radians(theta1)), l1],
                [0, 0, 0, 1]])

print('forward kinematics for l1:')
print(T01)

T12 = np.array([[np.cos(math.radians(theta2)), 0, np.sin(math.radians(theta2)), 0],
                [0, 1, 0, 0],
                [-np.sin(math.radians(theta2)), 0, np.cos(math.radians(theta2)), l2],
                [0, 0, 0, 1]])

print('forward kinematics for l2:')
print(T01 * T12)

T23 = np.array([[np.cos(math.radians(theta3)), 0, np.sin(math.radians(theta3)), 0],
                [0, 1, 0, 0],
                [-np.sin(math.radians(theta3)), 0, np.cos(math.radians(theta3)), l3],
                [0, 0, 0, 1]])

print('forward kinematics for l3:')
print(T01 * T12 * T23)

#starting positions for 2d example
x0 = 3
y0 = 2

# forward kinematics functions
x1 = x0 + l1 * np.cos(math.radians(theta1))
y1 = y0 + l1 * np.sin(math.radians(theta1))

x2 = x1 + l2 * np.cos(math.radians(theta1 + theta2))
y2 = y1 + l2 * np.sin(math.radians(theta1 + theta2))

x3 = x2 + l3 * np.cos(math.radians(theta1 + theta2 + theta3))
y3 = y2 + l3 * np.sin(math.radians(theta1 + theta2 + theta3))


#plot
plt.figure()
plt.plot([x0, x1], [y0, y1])
print('l1 coordinations:', [x0, y0], [x1, y1])
plt.plot([x1, x2], [y1, y2])
print('l2 coordinations:', [x1, y1], [x2, y2])
plt.plot([x2, x3], [y2, y3])
print('l3 coordinations:', [x2, y2], [x3, y3])
plt.xlim(2, 3.5)
plt.ylim(1, 2.5)
plt.grid()
plt.title('2d robotic leg')
plt.show()
