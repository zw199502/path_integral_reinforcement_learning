# -*- coding: UTF-8 -*-

import numpy as np
import time
import matplotlib.pyplot as pl

training_times = 100  # training times
roll_outs = 20  # path number
dt = 0.001  #step length of simulation, second
max_step = 120000  #maximum time steps

K_N = 4  # number of parapemers
N_convergence = 5000

class RL_PI2:
    def __init__(self):
        print(time.time())
        self.K = np.zeros((K_N, 1),dtype=np.float64)
        self.K_roll = np.zeros((K_N, roll_outs), dtype=np.float64)  #training parameters
        self.K_record = np.zeros((K_N, roll_outs, training_times), dtype=np.float64)  # record training parameters
        self.sigma = np.zeros((K_N, 1), dtype=np.float64)  #standard deviation of the variety about training parameters
        self.k_delta = np.zeros((K_N, roll_outs), dtype=np.float64)
        self.loss = np.zeros((roll_outs, 1), dtype=np.float64)  #loss function
        self.loss_record = np.zeros((roll_outs, training_times), dtype=np.float64)  #record loss function
        self.loss_after_training = np.zeros((training_times, 1), dtype=np.float64)  #loss function after each training
        self.K_after_training = np.zeros((K_N, training_times), dtype=np.float64)  # K after each training
        self.alpha = 0  # attenuation coefficient
        self.attenuation_step_length = 40  #sigma is attennuated every attenuation_step_length training times
        self.alpha = 0.85  #sigma is attennuated at 0.85 ratio
        self.current_roll = 0
        self.current_training = 0

        self.PI2_coefficient = 30.0 #PI2 coefficient

        ###system model
        # constants of crane model
        self.mp = 0.5
        self.mc = 3.5
        self.l = 0.9
        self.g = 9.8
        # control input
        self.F = 0
        self.F_max = 60

        # distance error
        self.my_ex = 0

        # variables of model
        self.m_theta = 0
        self.eta_theta = 0

        # swing angle
        self.theta = 0  # real time swing angle
        self.theta_dot = 0
        self.theta_dot2 = 0

        # initial state
        self.xd = 1 #target position, initial state needing changes
        # real time position
        self.xt = 0
        self.xt_dot = 0
        self.xt_dot2 = 0

        # initial control gain
        self.kp = 6
        self.kd = 1.25
        self.kE = 1.2
        self.kv = 3.4
        self.K[0] = self.kp
        self.K[1] = self.kd
        self.K[2] = self.kE
        self.K[3] = self.kv



        #initial PI2 parameters
        self.sigma[0] = 1.2
        self.sigma[1] = 0.25
        self.sigma[2] = 0.24
        self.sigma[3] = 0.68

    def training(self):
        for i in range(training_times):
            self.current_training = i

            # print(i)
            if i % self.attenuation_step_length == 0 and i!=0:
                self.sigma = self.sigma  * self.alpha  # attenuation
            for j in range(roll_outs):
                self.current_roll = j
                # print(j)
                self.k_delta[0, j] = np.random.normal(0, self.sigma[0], 1)
                self.k_delta[1, j] = np.random.normal(0, self.sigma[1], 1)
                self.k_delta[2, j] = np.random.normal(0, self.sigma[2], 1)
                self.k_delta[3, j] = np.random.normal(0, self.sigma[3], 1)
                self.K_roll[0, j] = self.K[0] + self.k_delta[0, j]
                self.K_roll[1, j] = self.K[1] + self.k_delta[1, j]
                self.K_roll[2, j] = self.K[2] + self.k_delta[2, j]
                self.K_roll[3, j] = self.K[3] + self.k_delta[3, j]
                self.loss[j] = self.model(self.K_roll[0, j], self.K_roll[1, j], self.K_roll[2, j], self.K_roll[3, j])
                print(i, j, self.loss[j])
                self.loss[j] = self.loss[j] + np.random.uniform(-0.001, 0.001, 1) #avoid the same loss


            # print(self.loss)
            self.K_record[:, :, self.current_training] = self.K_roll
            self.loss_record[:, self.current_training] = self.loss[:, 0]
            exponential_value_loss = np.zeros((roll_outs, 1), dtype=np.float64)  #
            probability_weighting = np.zeros((roll_outs, 1), dtype=np.float64)  # probability weighting of each roll
            for i2 in range(roll_outs):
                exponential_value_loss[i2] = np.exp(-self.PI2_coefficient * (self.loss[i2] - self.loss.min())
                                                   / (self.loss.max() - self.loss.min()))
            for i2 in range(roll_outs):
                probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)


            temp_k = np.dot(self.k_delta, probability_weighting)
            # print(self.sigma)

            #updata
            self.K = self.K + temp_k
            # print(self.K)
            self.K_after_training[:, self.current_training] = self.K[:, 0]
            self.loss_after_training[self.current_training] = self.model(self.K[0], self.K[1], self.K[2], self.K[3])

        pl.plot(self.loss_after_training)
        pl.show()
        np.savetxt('K_after_training_crane.txt', self.K_after_training)
        np.savetxt('loss_after_training_crane.txt', self.loss_after_training)
        print(time.time())

    def model(self, K0, K1, K2, K3):
        self.kp = K0
        self.kd = K1
        self.kE = K2
        self.kv = K3
        self.xt = 0
        self.xt_dot = 0
        self.xt_dot2 = 0
        self.theta = 0
        self.theta_dot = 0
        self.theta_dot2 = 0
        #if (ex < min_ex) keeps (N_convergence * dt)s, the result is converged

        # real time position
        t_x = np.zeros((max_step + 1, 1), dtype=np.float64)

        for tt in range(max_step):
            c_theta = np.cos(self.theta)
            s_theta = np.sin(self.theta)
            self.my_ex = self.xt - self.xd
            self.eta_theta = self.mp * s_theta * (self.l * np.power(self.theta_dot, 2) + self.g * c_theta)
            self.m_theta = self.mc + self.mp * np.power(s_theta, 2)
            self.F = (-self.kd * self.xt_dot - self.kp * self.my_ex + self.kv * (
                    self.eta_theta - self.mp * s_theta * c_theta * self.xt_dot * self.theta_dot)) / (self.kE + self.kv)
            self.xt_dot2 = self.F / self.m_theta - self.eta_theta / self.m_theta
            self.theta_dot2 = 1 / self.l * c_theta * self.xt_dot2 - self.g / self.l * s_theta
            #constraint
            if self.F > self.F_max:
                self.F = self.F_max

            if self.F < -self.F_max:
                self.F = -self.F_max
            #updata
            self.xt = self.xt + self.xt_dot * dt + 1 / 2 * self.xt_dot2 * dt * dt

            self.xt_dot = self.xt_dot + self.xt_dot2 * dt

            self.theta = self.theta + self.theta_dot * dt + 1 / 2 * self.theta_dot2 * dt * dt

            if self.theta > 5.0 / 180.0 * np.pi:
                return (2 * max_step * dt)

            self.theta_dot = self.theta_dot + self.theta_dot2 * dt

            t_x[tt + 1] = self.xt

        #Judging whether the system has arrived at the adjustment time

        for i_convergence in range(1, max_step):
            if (t_x[i_convergence] > t_x[i_convergence - 1]) and (t_x[i_convergence] > t_x[i_convergence + 1]) and (np.abs(t_x[i_convergence] - self.xd) <= 0.05):
                return (i_convergence * dt)

        return (max_step * dt)

if __name__ == '__main__':
    pi2 = RL_PI2()
    pi2.training()