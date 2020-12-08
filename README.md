Corralling Stochastic Bandit Algorithms code for experiments

To run an experiment call the run_experiment script with example input:
Corralling_Tsallis12 1000000 '[6,5,10]' '[0.5,0.2,0.01,0.19]' 5 1.

The first variable in the input is the corralling algorithm and takes values: "Corralling_Tsallis12"/"Corralling_LogBar"/"Corralling_UCB".

The second variable in the input is the time horizon T.

The third variable in the input is an array containing the number of total algorithms, the number of arms for sub-optimal algorithms and the number of arms for the best algorithm, respectively.

The fourth variable in the input is an array containing the average reward, the low reward, the intra-aglorithm gap and the inter-algorithm gap as described in Appendix 8.

The fifth variable is the number of times  the experiment is repeated.

The sixth variable denotes the number of current experiments.

The output of the script is going to be a file which is read by the jupyter notebook Plot_corral_exp.ipynb. The notebook will plot the respective file, if provided with its name.

To determine the order of corralled algorithms in the experiments change lines 97-107 in the run_experiments.py file.
