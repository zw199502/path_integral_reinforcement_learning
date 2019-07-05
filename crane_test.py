# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as pl


dt = 0.001  # step length of simulation, second
max_step = 120000  # maximum time steps


class crane_model:
    def __init__(self):
        self.t_current = np.zeros((max_step + 1, 1),dtype=np.float64)
        self.t_x = np.zeros((max_step + 1, 1),dtype=np.float64)
        self.t_theta = np.zeros((max_step + 1, 1), dtype=np.float64)
        self.t_F = np.zeros((max_step + 1, 1), dtype=np.float64)
        self.t_xd = np.zeros((max_step + 1, 1), dtype=np.float64)
        ###system model
        # constants of crane model
        self.mp = 0.5
        self.mc = 3.5
        self.l = 0.9
        self.F_max = 60
        self.g = 9.8
        # variables of model
        self.m_theta = 0
        self.eta_theta = 0

        # drift angle
        self.theta = 0
        self.theta_dot = 0
        self.theta_dot2 = 0

        # initial state
        self.xd = 1

        self.xt = 0

        self.xt_dot = 0

        self.xt_dot2 = 0

        self.K = np.loadtxt('K_after_training_crane_xd=1.txt')
        #control gain
        # self.kp = self.K[0, 90]
        # self.kd = self.K[1, 90]
        # self.kE = self.K[2, 90]
        # self.kv = self.K[3, 90]
        self.kp = 6
        self.kd = 1.25
        self.kE = 1.2
        self.kv = 3.4

    def training(self):
        self.model()

    def model(self):

        for tt in range(max_step):
            # self.xd = 1 - np.exp(-8.33 * np.power(self.t_current[tt], 3))
            # self.t_xd[tt + 1] = self.xd
            c_theta = np.cos(self.theta)
            s_theta = np.sin(self.theta)
            my_ex = self.xt - self.xd
            self.eta_theta = self.mp * s_theta * (self.l * np.power(self.theta_dot, 2) + self.g * c_theta)
            self.m_theta = self.mc + self.mp * np.power(s_theta, 2)
            # M = np.zeros((2, 2), dtype=np.float64)
            # M[0, 0] = self.mc + self.mp
            # M[0, 1] = -self.mp * self.l * c_theta
            # M[1, 0] = -self.mp * self.l * c_theta
            # M[1, 1] = self.mp * np.power(self.l ,2)
            # q_dot = np.zeros((2, 1),dtype=np.float64)
            # q_dot[0] = self.xt_dot
            # q_dot[1] = self.theta_dot
            # E = 1 / 2 * np.dot(np.dot(q_dot.T, M), q_dot) + self.mp * self.g * self.l * (1 - c_theta)
            # F = (-self.kd * self.xt_dot - self.kp * my_ex + self.kv * self.eta_theta / self.m_theta) / (self.kE * E + \
            #     self.kv / self.m_theta)
            F = (-self.kd * self.xt_dot - self.kp * my_ex + self.kv * (self.eta_theta - self.mp * s_theta * c_theta * self.xt_dot * self.theta_dot)) / (self.kE + self.kv)
            self.xt_dot2 = F / self.m_theta - self.eta_theta / self.m_theta
            self.theta_dot2 = 1 / self.l * c_theta * self.xt_dot2 - self.g / self.l * s_theta

            ##updata
            self.xt = self.xt + self.xt_dot * dt + 1 / 2 * self.xt_dot2 * dt * dt

            self.xt_dot = self.xt_dot + self.xt_dot2 * dt

            self.theta = self.theta + self.theta_dot * dt + 1 / 2 * self.theta_dot2 * dt * dt

            self.theta_dot = self.theta_dot + self.theta_dot2 * dt

            self.t_x[tt + 1] = self.xt
            self.t_theta[tt + 1] = self.theta
            self.t_current[tt + 1] = (tt + 1) * dt
            self.t_F[tt + 1] = F

        pl.plot(self.t_current, self.t_x)
        pl.show()
        # np.savetxt('t_current.txt',self.t_current)
        np.savetxt('t_x.txt', self.t_x)
        np.savetxt('t_theta.txt', self.t_theta)


if __name__ == '__main__':
    crane = crane_model()
    crane.training()