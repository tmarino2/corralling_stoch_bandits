import sys
import io
import random
import numpy as np
from scipy import optimize
import math
import copy
import json
import pickle
import importlib

import UCB_I
import Tsallis_INF
import Thompson_Sampling

def run_experiment(total_rounds, rounds_index, corralling_alg, sub_algs, sub_algs_rewards, Corral_stoch, num_run='0'):
    corral_instance = Corral_stoch(corralling_alg, sub_algs, sub_algs_rewards)
    regrets = []
    alg_pulls = []
    corral_distr = []
    rewards_arr = list(json.loads(sys.argv[4]))
    alg_params = list(json.loads(sys.argv[3]))
    num_algs = int(alg_params[0])
    num_arms = [int(alg_params[1]) for i in range(num_algs)]
    num_arms[0] = int(alg_params[2])
    
    for t in range(total_rounds):
        corral_instance.play_round()
        if(t in rounds_index):
            regrets.append(copy.copy(corral_instance.regret_t))
            alg_pulls.append(copy.copy(corral_instance.num_alg_pulls))
            try:
                corral_distr.append(copy.copy(corralling_alg.prob_distr))
            except: 
                corral_distr.append(copy.copy(corral_instance.num_alg_pulls/sum(corral_instance.num_alg_pulls)))
    fname = './pickle_dumps/Corral_epochs_{0}_params_{1}_corraltype_{2}_numrun_{3}'.format(total_rounds,str(sys.argv[4]),str(sys.argv[1]), num_run)
    curr_run_dict = {'regrets': regrets, 'corral_distr': corral_distr, 'corral_instance': corral_instance,\
                     'rounds_index': rounds_index, 'alg_rewards':sub_algs_rewards,\
                     'rewards_params':[float(rewards_arr[0]),float(rewards_arr[1]),float(rewards_arr[2]),float(rewards_arr[3])],\
                     'num_arms':num_arms, 'alg_pulls':alg_pulls}
    with io.open(fname,'ab+') as dump_file:
        pickle.dump(curr_run_dict, dump_file)
    return (regrets,alg_pulls,corral_distr,corral_instance)

corralling_type_str = str(sys.argv[1])
T = int(sys.argv[2])
rounds_index = [t for t in range(T) if t%50==0]
alg_params = list(json.loads(sys.argv[3])) #num_algs, num_arms, num_arms[0]
rewards_arr = list(json.loads(sys.argv[4])) #std_reward, low_reward, in_alg_gap, out_alg_gap,
number_of_runs = int(sys.argv[5])
exp_num = int(sys.argv[6])

num_algs = int(alg_params[0])
num_arms = [int(alg_params[1]) for i in range(num_algs)]
num_arms[0] = int(alg_params[2])

std_reward = float(rewards_arr[0])
low_reward = float(rewards_arr[1])
in_alg_gap = float(rewards_arr[2])
out_alg_gap = float(rewards_arr[3])

for run in range(number_of_runs):
    beta = math.exp(1/math.log(T)) #multiplicative param for step-size for OMD type corralling
    delta = 1.0/math.pow(T,2) #confidence interval param for UCB type corralling
    delta2 = 0.01 #confidence interval param only for UCB corralling and UCBs
    eta = np.ones(num_algs)/math.sqrt(T) #fixed step-size only for LogBar corralling
    
    init_rewards = [np.ones(num_arms[i]) for i in range(num_algs)]
    sub_algs = []
    for i in range(num_algs):
        if i < np.floor(num_algs/3):
            sub_algs.append(UCB_I.UCB_I(num_arms[i], init_rewards[i], delta))
        elif i < 2*np.floor(num_algs/3):
            sub_algs.append(Tsallis_INF.Tsallis_INF(num_arms[i]))
        else:
            sub_algs.append(Thompson_Sampling.Thompson_Sampling(num_arms[i]))
    alg_rewards = {sub_algs[i]: std_reward*np.ones(num_arms[i]) for i in range(num_algs)}
    for alg in sub_algs:
        alg_rewards[alg][0] += in_alg_gap
        alg_rewards[sub_algs[0]] = low_reward*np.ones(num_arms[0])
        alg_rewards[sub_algs[0]][0] = std_reward+ in_alg_gap+  out_alg_gap
        init_thresh = 5*num_algs
        thresholds = [init_thresh*math.pow(2,i) for i in range(math.floor(math.log2(T)/2))]

    #Only for UCB_corralling
    algs = {}

    Corral_stoch = None
    corralling_alg = None
    corralling_type = None

    if corralling_type_str == 'Corralling_UCB':
        corralling_type = importlib.import_module(corralling_type_str)
        Corral_stoch = corralling_type.Corral_stoch
        corralling_alg = corralling_type.UCB_C(num_arms, delta2)
        for i in range(num_algs):
            alg = sub_algs[i]
            algs[alg] = [alg]
            if i < np.floor(num_algs/3):
                for j in range(int(np.ceil(np.log(1 + 1.0/delta2)))-1):
                    algs[alg].append(UCB_I.UCB_I(num_arms[i], init_rewards[i], delta))
            elif i < 2*np.floor(num_algs/3):
                for j in range(int(np.ceil(np.log(1 + 1.0/delta2)))-1):
                    algs[alg].append(Tsallis_INF.Tsallis_INF(num_arms[i]))
            else:
                for j in range(int(np.ceil(np.log(1 + 1.0/delta2)))-1):
                    algs[alg].append(Thompson_Sampling.Thompson_Sampling(num_arms[i]))
        sub_algs = algs
    elif corralling_type_str == 'Corralling_Tsallis12':
        corralling_type = importlib.import_module(corralling_type_str)
        Corral_stoch = corralling_type.Corral_stoch
        corralling_alg = corralling_type.Tsallis_C(num_algs, thresholds, beta)
    elif corralling_type_str == 'Corralling_LogBar':
        corralling_type = importlib.import_module(corralling_type_str)
        Corral_stoch = corralling_type.Corral_stoch
        corralling_alg = corralling_type.LogBarCorral(num_algs, thresholds, beta, eta)
    else:
        assert False, "Unkown corralling type"

    (regrets,alg_pulls,corral_distr,corral_instance) = run_experiment(T,rounds_index, corralling_alg, sub_algs ,alg_rewards, Corral_stoch, exp_num)
