# -*- coding: UTF-8 -*-

import matplotlib.pyplot as pl
import numpy as np

N = 450

dt = 0.035

x2 = -1.5
y2 = -0.6
theta2 = -55.0 / 180.0 * np.pi

k_d1 = 4.5
k_theta1 = 3.2

x_target = np.zeros(N + 1, dtype=np.float64)
y_target = np.zeros(N + 1, dtype=np.float64)
theta_target = np.zeros(N + 1, dtype=np.float64)

x = np.zeros(N + 1, dtype=np.float64)
y = np.zeros(N + 1, dtype=np.float64)
theta = np.zeros(N + 1, dtype=np.float64)

t_current = np.zeros(N + 1, dtype=np.float64)

x[0] = x2
y[0] = y2
theta[0] =theta2

x_t = x2
y_t = y2
theta_t = theta2

K1 = 1.15
K2 = 0.65

u = 0.22
r = 0



r_last = 0
max_r = 1.58
max_r_delta = 2.06


def tuning_k(d_error):
    k_d1 = 3.4
    k_theta1 = 2.2
    if d_error >= 0.8 or d_error <= -0.8:
        k_d1 = 3.4
        k_theta1 = 2.2

    if (d_error < 0.8 and d_error >= 0.7) or (d_error > -0.8 and d_error <= -0.7):
        k_d1 = 4.0
        k_theta1 = 2.9

    if (d_error < 0.7 and d_error >= 0.6) or (d_error > -0.7 and d_error <= -0.6):
        k_d1 = 4.5
        k_theta1 = 3.2

    if (d_error < 0.6 and d_error >= 0.5) or (d_error > -0.6 and d_error <= -0.5):
        k_d1 = 5.0
        k_theta1 = 3.4

    if (d_error < 0.5 and d_error >= 0.4) or (d_error > -0.5 and d_error <= -0.4):
        k_d1 = 5.5
        k_theta1 = 3.8

    if (d_error < 0.4 and d_error >= 0.3) or (d_error > -0.4 and d_error <= -0.3):
        k_d1 = 5.8
        k_theta1 = 4.2

    if (d_error < 0.3 and d_error >= 0.2) or (d_error > -0.3 and d_error <= -0.2):
        k_d1 = 6.6
        k_theta1 = 4.5

    if (d_error < 0.2 and d_error >= 0.1) or (d_error > -0.2 and d_error <= -0.1):
        k_d1 = 7.6
        k_theta1 = 7.0

    if (d_error < 0.1 and d_error >= 0.08) or (d_error > -0.1 and d_error <= -0.08):
        k_d1 = 8.6
        k_theta1 = 8.0

    if (d_error < 0.08 and d_error >= 0.06) or (d_error > -0.08 and d_error <= -0.06):
        k_d1 = 10
        k_theta1 = 8.6

    if (d_error < 0.06 and d_error >= 0.04) or (d_error > -0.06 and d_error <= -0.04):
        k_d1 = 11
        k_theta1 = 10.0

    if (d_error < 0.04 and d_error >= 0.02) or (d_error > -0.04 and d_error <= -0.02):
        k_d1 = 19
        k_theta1 = 12.0

    if d_error < 0.02 and d_error > -0.02:
        k_d1 = 25
        k_theta1 = 16.0

    return k_d1,k_theta1

for tt in range (N):
    e_theta = theta_t
    e_d = y_t

    if tt % 10 == 0:
        k_d1, k_theta1 = tuning_k(e_d)
    print(k_d1, k_theta1)

    e_theta1 = -k_d1 * e_d - theta_t


    r = k_theta1 * e_theta1
    if r > max_r:
        r = max_r
    if r - r_last > max_r_delta:
        r = r_last + max_r_delta
    if r < -max_r:
        r = -max_r
    if r - r_last < -max_r_delta:
        r = r_last - max_r_delta
    r_last = r
    x_t = x_t + u * np.cos(theta_t) * dt
    y_t = y_t + u * np.sin(theta_t) * dt
    theta_t = theta_t + r * dt
    x[tt + 1]=x_t
    y[tt + 1]=y_t
    theta[tt + 1] = theta_t
    x_target[tt + 1]=x_t
    y_target[tt + 1]=0.0
    t_current[tt + 1] = (tt + 1) * dt


# print(x)
pl.subplot(211)
pl.plot(t_current,y,'r',t_current,y_target)
pl.subplot(212)
pl.plot(t_current,theta,'r',t_current,theta_target)
pl.show()

np.savetxt('t_current_tuning.txt',t_current)
np.savetxt('d_tuning.txt',y)
np.savetxt('theta_tuning.txt',theta)





