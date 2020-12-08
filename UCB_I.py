import sys
import io
import random
import numpy as np
from scipy import optimize
import math
import matplotlib.pyplot as plt
import copy
import json
import pickle

class UCB_I(object):
    def __init__(self, num_arms, init_rewards, delta):
        self.num_arms = num_arms
        self.conf_bounds = np.zeros(num_arms)
        self.mean_est = np.zeros(num_arms)
        self.num_pulls = np.zeros(num_arms)
        self.delta = delta
        self.arm_t = -1
        for i in range(num_arms):
            self.mean_est[i] += init_rewards[i]
            self.num_pulls[i] += 1
            self.conf_bounds[i] = math.sqrt(2.0*math.log(1.0/delta)/self.num_pulls[i])   
        
    def play_arm(self):
        UCB_est = np.array([mu_t + ucb_bound for mu_t, ucb_bound in zip(self.mean_est, self.conf_bounds)])
        self.arm_t = np.argmax(UCB_est)
        return np.argmax(UCB_est)
    
    def update_distr(self,reward_t):
        a_t = self.arm_t
        self.mean_est[a_t] = (self.mean_est[a_t]*self.num_pulls[a_t] + reward_t)/(1.0*self.num_pulls[a_t]+1.0)
        self.num_pulls[a_t] += 1
        self.conf_bounds[a_t] =  math.sqrt(2.0*math.log(1.0/self.delta)/self.num_pulls[a_t])
