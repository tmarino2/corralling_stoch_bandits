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

def NewtonLogBar(x, ell_t, w_old, eta, thresh, max_iter=1e5):
    curr_iter = 0
    w_t = 1/(1/w_old + eta*(ell_t - x))
    pot_deriv = 1/(1/w_old + eta*(ell_t - x))**2
    while(abs(np.sum(w_t)-1)>thresh and curr_iter <= max_iter):
        curr_iter +=1
        x = x - (np.sum(w_t)-1)/np.sum(pot_deriv)
        w_t = 1/(1/w_old + eta*(ell_t - x))
    return x

class LogBarCorral(object):
    def __init__(self, num_algs, thresholds, beta, eta, step_decay_rate = 0.5, alpha = 0.5):
        self.alpha = alpha
        self.inverse_exponent = 1.0 / (self.alpha - 1.0)
        self.num_algs = num_algs
        self.thresholds = thresholds
        self.beta = beta
        self.step_decay_rate = step_decay_rate
        self.alg_thresholds_indx = [0 for i in range(num_algs)] #indexes which is the next threshold on which we double the step-size
        self.prob_distr = np.ones(num_algs)/(1.0*num_algs)
        self.alg_t = 0
        self.loss_vec_t = np.zeros(num_algs)
        self.est_loss_t = np.zeros(num_algs)
        self.round = self.num_algs
        self.eta_t = eta
        self.eta_base = eta
        self.eta_t_mult_factor = np.zeros(num_algs) #the step size will be formed by the Hadamart product of eta_t and eta_t_mult_factor
        self.x_t = 1 #normalization cosntant for Tsallis-INF
        self.restart_round = 10*self.num_algs
        
    def play_alg(self):
        if(self.round%50000==0):
            print("prob_distr at sampling "+str(self.prob_distr))
            print("losses "+str(self.loss_vec_t))
            print("step size "+str(self.eta_t)+" \n\n")
        play_distr = copy.copy(self.prob_distr)
        if(self.round > self.num_algs):
            play_distr = (1-1.0/self.round)*play_distr + 1.0/self.round*np.array([1.0/self.num_algs for i in range(self.num_algs)])
        self.alg_t = np.random.choice(self.num_algs, 1, p=list(play_distr))[0]
        self.round += 1
        return self.alg_t
    
    def normal_const(self,x):
        return sum(1/(1/self.prob_distr + self.eta_t*(self.est_loss_t - x)))-1
    
    def update_distr(self, loss_t):
        loss_t = 1-loss_t #working with rewards
        a_t = self.alg_t
        self.est_loss_t = np.zeros(self.num_algs)
        self.est_loss_t[a_t] = loss_t/self.prob_distr[a_t]
        if(any(self.est_loss_t) <0):
            print(loss_t)
            print(self.prob_distr[a_t])
        self.loss_vec_t += self.est_loss_t
        try:
#             self.x_t = NewtonLogBar(self.x_t, self.est_loss_t, self.prob_distr, self.eta_t, 1e-4, max_iter=1e5)
            self.x_t = optimize.newton(self.normal_const, self.x_t, tol=1e-4)
        except RuntimeError:
            print("round "+str(self.round)+" x_t "+ str(self.x_t))
            print("loss vector "+str(self.loss_vec_t))
            print("step sizes "+str(self.eta_t))
        self.prob_distr = 1/(1/self.prob_distr + self.eta_t*(self.est_loss_t - self.x_t))
        self.prob_distr /= np.sum(self.prob_distr)
        shift_index = False #indicates if we have shifted the losses yet before applying the step-size rescaling for thresholds
        for i in range(self.prob_distr.shape[0]):
            self.eta_t[i] = math.pow(self.beta, self.eta_t_mult_factor[i])*self.eta_base[i]
            try:
                if(len(self.thresholds) > self.alg_thresholds_indx[i] and self.prob_distr[i] < 1/self.thresholds[self.alg_thresholds_indx[i]]):
                    self.eta_t[i] = math.pow(self.beta, self.eta_t_mult_factor[i])*self.eta_base[i]
            except IndexError:
                print("prob distr shape "+str(self.prob_distr.shape))
                print("len of alg_thresholds_indx "+str(len(self.alg_thresholds_indx)))
                print("len of threshold "+str(len(self.thresholds)))
                print("i "+str(i))
                print("self.alg_thresholds_indx[i] "+ str(self.alg_thresholds_indx[i]))
                break

class Corral_stoch(object):    
    def __init__(self, corralling_alg, sub_algs, sub_algs_rewards = {}):
        '''corralling_alg is an instance of a corralling algorithm with a method pull_arm 
        returning an index of sub alg,
            sub_algs is a list of isntances of algorithms to be corralled like UCB_I,
            sub_algs_rewards is a dictionary with keys the sub algorithms and items lists of Ber parameters'''
        self.corralling_alg = corralling_alg
        self.sub_algs = sub_algs
        self.sub_algs_rewards = sub_algs_rewards
        self.best_arms = {}
        self.num_alg_pulls = np.zeros(len(sub_algs))
        self.best_alg = None
        self.best_arm_val = -1
        self.regret_t = 0
        self.curr_round = 0
        self.gen_rand_rewards = False
        self.total_ell_t = 0
        self.num_algs = len(sub_algs)
        
        if(len(sub_algs_rewards.keys())==0):
            self.gen_rand_rewards = True
        if(self.gen_rand_rewards==True):
            for alg in self.sub_algs:
                num_arms = alg.num_arms
                alg_reward = np.random.rand(num_arms)
                self.sub_algs_rewards[alg] = alg_reward
        for alg in self.sub_algs:
            best_alg_arm = np.argmax(self.sub_algs_rewards[alg])
            self.best_arms[alg] = best_alg_arm
            curr_alg_rewards = self.sub_algs_rewards[alg]
            if(self.best_arm_val < curr_alg_rewards[best_alg_arm]):
                self.best_arm_val = curr_alg_rewards[best_alg_arm]
                self.best_alg = alg       
                
    def play_round(self):
        self.curr_round += 1
        selected_alg_t_index = self.corralling_alg.play_alg() #select algorithm to play from master
        selected_alg_t = self.sub_algs[selected_alg_t_index]
        self.num_alg_pulls[selected_alg_t_index] += 1
        selected_arm_t = selected_alg_t.play_arm() #select arm to play from sub algorithm 
        reward_t = np.random.binomial(1,(self.sub_algs_rewards[selected_alg_t])[selected_arm_t],1)[0] #reward of arm
        hat_reward_t = np.zeros(self.num_algs) #construct unbiased estimators of rewards
        hat_reward_t[selected_alg_t_index] = reward_t/self.corralling_alg.prob_distr[selected_alg_t_index]
        for i in range(self.num_algs): #update each sub algorithm
            self.sub_algs[i].update_distr(hat_reward_t[i])
        self.regret_t += self.best_arm_val - reward_t
        self.total_ell_t += (1-reward_t)
        self.corralling_alg.update_distr(reward_t)





