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

class UCB_C(object):
    def __init__(self, alg_compls, delta):
        self.curr_round = 0.0
        self.num_algs =len(alg_compls)
        self.alg_compls = alg_compls #R(T)/sqrt(T\log(T)) -- essentially number of arms
        self.delta = delta
        self.avr_rewards = np.zeros((self.num_algs,int(np.ceil(np.log(1 + 1.0/self.delta))))) #num_algs \times 1.0/delta array cotanining the average rewards
        self.algs_ucbs = np.zeros(self.num_algs) #the UCBs for each of the algs
        self.num_alg_pulls = np.zeros(self.num_algs)
        self.alg_t = 0
        for i in range(self.num_algs):
            self.algs_ucbs[i] = 1e6
    
    def play_alg(self):
        max_ucb = 0.0
        for i in range(self.num_algs):
            alg_rewards = self.avr_rewards[i]
            alg_median = np.percentile(alg_rewards,51,interpolation='nearest') #take median
            if max_ucb < alg_median + self.algs_ucbs[i]: #select algorithm with highest median + ucb
                max_ucb = alg_median + self.algs_ucbs[i]
                self.alg_t = i
        self.num_alg_pulls[self.alg_t]+=1
        return self.alg_t
            
    def update_distr(self, rewards):
        assert(len(rewards) == int(np.ceil(np.log(1 + 1.0/self.delta)))), "Length of rewards does not match number of algorithm instances."
        num_pulls = self.num_alg_pulls[self.alg_t]
        self.avr_rewards[self.alg_t] = ((num_pulls-1)/(num_pulls))*self.avr_rewards[self.alg_t] + rewards/(num_pulls) #update all of the instances of selected algorithm
        self.curr_round+=1
        for i in range(self.num_algs):
            if(self.num_alg_pulls[i] >0):
                self.algs_ucbs[i] = 8.0*math.sqrt(self.alg_compls[i]*math.log(self.curr_round+1))/(self.num_alg_pulls[i]) #update ucb for selected algorithm
        if self.curr_round%10000==0:
            print(self.num_alg_pulls)
            print(str(self.algs_ucbs)+"+\n"+str(self.avr_rewards)+"\n\n")

class Corral_stoch():
    def __init__(self, corralling_alg, sub_algs, sub_algs_rewards = {}):
        '''corralling_alg is an instance of a corralling algorithm with a method pull_arm 
        returning an index of sub alg,
            sub_algs is a dictonary of log(1/\delta) isntances of algorithms to be corralled like UCB_I with key an instance of the alg,
            sub_algs_rewards is a dictionary with keys the sub algorithms and items lists of Ber parameters'''
        self.corralling_alg = corralling_alg
        self.sub_algs = sub_algs #dictionary with keys an instance of an algorithm to be corralled and entries log(1/delta) copies of said algorithm
        self.sub_algs_list = list(sub_algs.keys()) #the corralled algs keys to be used for idnexing 
        self.sub_algs_rewards = sub_algs_rewards
        self.best_arms = {}
        self.num_alg_pulls = np.zeros(len(self.sub_algs_list))
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
        selected_alg_t = self.sub_algs_list[selected_alg_t_index]
        self.num_alg_pulls[selected_alg_t_index] += 1
        rewards_array_t = [] #make sure to convert to np.array
        for sub_alg_t in self.sub_algs[selected_alg_t]:
            selected_arm_t = sub_alg_t.play_arm() #select arm to play from sub algorithm 
            reward_t = np.random.binomial(1,(self.sub_algs_rewards[selected_alg_t])[selected_arm_t],1)[0] #reward of arm
            rewards_array_t.append(reward_t)
            sub_alg_t.update_distr(reward_t) #update internal state of alg
            self.regret_t += self.best_arm_val - reward_t
            self.total_ell_t += (1-reward_t)
        self.corralling_alg.update_distr(np.array(rewards_array_t))
