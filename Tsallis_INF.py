import sys
import random
import math
import numpy as np
from scipy import optimize
import copy

def NewtonTsallis12(x, loss_vec, eta, thresh, max_iter=1e5):
    curr_iter = 0
    min_loss_vec = min(loss_vec)
    loss_vec = loss_vec - min_loss_vec
    w_t = 4*(eta*(loss_vec - x))**(-2)
    while(abs(np.sum(w_t)-1)>thresh and curr_iter <= max_iter):
        curr_iter +=1
        x = x - (np.sum(w_t)-1)/np.sum((w_t**(3./2.))*eta)
        w_t = 4*(eta*(loss_vec - x))**(-2)
    return x + min_loss_vec

class Tsallis_INF(object):
    def __init__(self, num_arms, alpha=0.5):
        self.alpha = alpha
        self.num_arms = num_arms
        self.prob_distr = np.ones(num_arms)/(1.0*num_arms)
        self.arm_t = 0
        self.loss_vec_t = np.zeros(num_arms)
        self.round = self.num_arms
        self.eta_t = 1
        self.inverse_exponent = 1.0 / (self.alpha - 1.0)
        self.x_t = -math.sqrt(num_arms) #normalization cosntant for Tsallis-INF
        
    def play_arm(self):
        self.arm_t = np.random.choice(self.num_arms, 1, p=list(self.prob_distr))[0]
        self.round += 1
        return self.arm_t
    
    def normal_const(self,x):
        res = 0.0
        for i in range(self.prob_distr.shape[0]):
            res += 4.0*math.pow(self.eta_t*(self.loss_vec_t[i] - min(self.loss_vec_t) - x),-2)
        return 1.0 - res
    
    def normal_const2(self,x):
        return (np.sum( (self.eta_t * (self.loss_vec_t - x)) ** self.inverse_exponent ) - 1) ** 2
    
    def update_distr(self,loss_t, eta_t=None):
        loss_t = 1-loss_t #working with rewards
        if eta_t is None:
            self.eta_t = 1/math.sqrt(self.round)
        else:
            self.eta_t = eta_t
        a_t = self.arm_t
        if(any(self.prob_distr)<1e-6):
            self.prob_distr = np.ones(num_arms)/(1.0*self.num_arms)
        est_loss_t = loss_t/self.prob_distr[a_t]
        self.loss_vec_t[a_t] += est_loss_t
        self.x_t = optimize.newton(self.normal_const, self.x_t)
#        self.x_t = NewtonTsallis12(-math.sqrt(self.num_arms), self.loss_vec_t, self.eta_t, 1e-10)
        for i in range(self.prob_distr.shape[0]):
            self.prob_distr[i] = 4.0*math.pow(self.eta_t*(self.loss_vec_t[i] - self.x_t - min(self.loss_vec_t)),-2)
        self.prob_distr /= np.sum(self.prob_distr)
