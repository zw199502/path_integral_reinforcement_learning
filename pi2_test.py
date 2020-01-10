# -*- coding: UTF-8 -*-

import matplotlib.pyplot as pl
import numpy as np

N = 450

k_all = np.loadtxt('K_all.txt')
x1 = 0
y1 = 0
theta1 = 0

x2 = -1.5
y2 = -0.6
theta2 = -55.0 / 180.0 * np.pi



x_target = np.zeros(N + 1, dtype=np.float64)
y_target = np.zeros(N + 1, dtype=np.float64)
theta_target = np.zeros(N + 1, dtype=np.float64)



x = np.zeros(N + 1, dtype=np.float64)
y = np.zeros(N + 1, dtype=np.float64)
theta = np.zeros(N + 1, dtype=np.float64)



x_t = x2
y_t = y2
theta_t = theta2

x_target[0] = x2
y_target[0] = 0

x[0] = x2
y[0] = y2
theta[0] = theta2

index_d = 0
index_phi = 0

K1 = 75.3
K2 = 14.8

u = 0.22
u_real = np.zeros(N, dtype=np.float64)
d_real = np.zeros(N, dtype=np.float64)
phi_real = np.zeros(N, dtype=np.float64)
run_time = np.zeros(N + 1, dtype=np.float64)
r_last = 0
max_r = 1.58
max_r_delta = 0.5

dt = 0.035

K1_ref = 50.0
K2_ref = 20.20
d_ref = 0
phi_ref = 0
r_ref = 0
r_last_ref = 0
x_ref = np.zeros(N + 1, dtype=np.float64)
y_ref = np.zeros(N + 1, dtype=np.float64)
theta_ref = np.zeros(N + 1, dtype=np.float64)
x_ref[0] = x2
y_ref[0] = y2
theta_ref[0] = theta2
x_t_ref = x2
y_t_ref = y2
theta_t_ref = theta2

#
K1_ref2 = 98.253
K2_ref2 = 14.447
d_ref2 = 0
phi_ref2 = 0
r_ref2 = 0
r_last_ref2 = 0
x_ref2 = np.zeros(N + 1, dtype=np.float64)
y_ref2 = np.zeros(N + 1, dtype=np.float64)
theta_ref2 = np.zeros(N + 1, dtype=np.float64)
x_ref2[0] = x2
y_ref2[0] = y2
theta_ref2[0] = theta2
x_t_ref2 = x2
y_t_ref2 = y2
theta_t_ref2 = theta2

for tt in range (N):

    d = y_t
    phi = theta_t

    if d < -0.795:
        index_d = 0
    if d > 0.795:
        index_d = 160
    if phi < -82.5 / 180.0 * np.pi:
        index_phi = 0
    if phi > 82.5 / 180.0 * np.pi:
        index_phi = 34
    if d >= -0.795 and d <= 0.795:
        for index in range(159):
            if d >= -0.795 + index * 0.01 and d < -0.795 + (index + 1) * 0.01:
                index_d = index + 1
                break
    if (phi >= -82.5 / 180.0 * np.pi) and (phi <= 82.5 / 180.0 * np.pi):
        for index in range(33):
            if (phi >= -82.5 / 180.0 * np.pi + index * 5 / 180.0 * np.pi) and (phi < -82.5 / 180.0 * np.pi + (index + 1) * 5 / 180.0 * np.pi):
                index_phi = index + 1
                break
    if tt % 10 == 0:
        K1 = k_all[index_d * 35 + index_phi, 0]
        K2 = k_all[index_d * 35 + index_phi, 1]
    d_real[tt] = d
    phi_real[tt] = phi

    # temp_u = u + np.random.uniform(-0.2, 0.2, 1)
    temp_u = u
    r = -K1 * temp_u * d * np.sin(phi) / phi - K2 * temp_u * phi
    if r > max_r:
        r = max_r
    if r - r_last > max_r_delta:
        r = r_last + max_r_delta
    if r < -max_r:
        r = -max_r
    if r - r_last < -max_r_delta:
        r = r_last - max_r_delta
    r_last = r

    print(K1, K2)
    x_t = x_t + temp_u * np.cos(theta_t) * dt
    y_t = y_t + temp_u * np.sin(theta_t) * dt
    theta_t = theta_t + r * dt

    x[1 + tt] = x_t
    y[1 + tt] = y_t
    theta[1 + tt] = theta_t

    d_ref = y_t_ref
    phi_ref = theta_t_ref

    # temp_u = u + np.random.uniform(-0.2, 0.2, 1)
    temp_u = u
    r_ref = -K1_ref * temp_u * d_ref * np.sin(phi_ref) / phi_ref - K2_ref * temp_u * phi_ref
    if r_ref > max_r:
        r_ref = max_r
    if r_ref - r_last > max_r_delta:
        r_ref = r_last + max_r_delta
    if r_ref < -max_r:
        r_ref = -max_r
    if r_ref - r_last < -max_r_delta:
        r_ref = r_last - max_r_delta
    r_last_ref = r_ref


    x_t_ref = x_t_ref + temp_u * np.cos(theta_t_ref) * dt
    y_t_ref = y_t_ref + temp_u * np.sin(theta_t_ref) * dt
    theta_t_ref = theta_t_ref + r_ref * dt

    x_ref[1 + tt] = x_t_ref
    y_ref[1 + tt] = y_t_ref
    theta_ref[1 + tt] = theta_t_ref

    d_ref2 = y_t_ref2
    phi_ref2 = theta_t_ref2

    # temp_u = u + np.random.uniform(-0.2, 0.2, 1)
    temp_u = u
    r_ref2 = -K1_ref2 * temp_u * d_ref2 * np.sin(phi_ref2) / phi_ref2 - K2_ref2 * temp_u * phi_ref2
    if r_ref2 > max_r:
        r_ref2 = max_r
    if r_ref2 - r_last > max_r_delta:
        r_ref2 = r_last + max_r_delta
    if r_ref2 < -max_r:
        r_ref2 = -max_r
    if r_ref2 - r_last < -max_r_delta:
        r_ref2 = r_last - max_r_delta
    r_last_ref2 = r_ref2

    x_t_ref2 = x_t_ref2 + temp_u * np.cos(theta_t_ref2) * dt
    y_t_ref2 = y_t_ref2 + temp_u * np.sin(theta_t_ref2) * dt
    theta_t_ref2 = theta_t_ref2 + r_ref2 * dt

    x_ref2[1 + tt] = x_t_ref2
    y_ref2[1 + tt] = y_t_ref2
    theta_ref2[1 + tt] = theta_t_ref2

    x_target[1 + tt] = x_t
    y_target[1 + tt] = 0

    run_time[tt + 1] = dt * (tt + 1)



# pl.plot(x, y, 'b',x_ref, y_ref, 'g', x_ref2, y_ref2, 'c', x_target, y_target, 'r')
# pl.plot(run_time, y, 'b',run_time, y_ref, 'g', run_time, y_ref2, 'c', run_time, y_target, 'r')
pl.plot(run_time, y)
# pl.plot(d_real)
pl.show()
# np.savetxt('x.txt',x)
# np.savetxt('y.txt',y)
# np.savetxt('theta.txt',theta)
#
# np.savetxt('x_target.txt',x_target)
# np.savetxt('y_target.txt',y_target)
# np.savetxt('u_real.txt',u_real)

np.savetxt('d_real.txt',y)
np.savetxt('phi_real.txt',theta)
np.savetxt('run_time.txt',run_time)

np.savetxt('d_ref.txt',y_ref)
np.savetxt('phi_ref.txt',theta_ref)
np.savetxt('d_ref2.txt',y_ref2)
np.savetxt('phi_ref2.txt',theta_ref2)





