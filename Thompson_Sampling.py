import sys
import random
import numpy as np

class Thompson_Sampling():
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_t = -1
        self.curr_round = 1
        self.alpha = np.ones(num_arms) #posterior param
        self.beta = np.ones(num_arms) #posterior param
        self.theta = np.zeros(num_arms)
    
    def play_arm(self):
        self.theta = np.random.beta(self.alpha,self.beta)
        try:
            self.arm_t = np.argmax(self.theta)[0]
        except IndexError:
            self.arm_t = np.argmax(self.theta)
        return self.arm_t
    
    def update_distr(self, r_t):
        self.curr_round+=1
        self.alpha[self.arm_t] += r_t
        self.beta[self.arm_t] = max(1,self.beta[self.arm_t] + 1 - r_t)
