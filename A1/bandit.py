import argparse
from epsilongreedy import epsilonGreedy
from ucb import Ucb
from KLucb import KLUCB
from thompsonsampling import thompsonSampling
import numpy as np
from thompsonsamplinghint import thompsonSamplinghint
import time
if __name__=="__main__":
    t=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument("--instance",help="instance is instance file")
    parser.add_argument("--algorithm",help="algorithm of epsilon-greedy, ucb, kl-ucb, thompson-sampling, or thompson-sampling-with-hint")
    parser.add_argument("--randomSeed",help="randomseed where is a non-negative integer")
    parser.add_argument("--epsilon",help="epsilon is where a number in [0, 1]")
    parser.add_argument("--horizon",help="horizon is where hz is a non-negative integer")
    epsilon=0.02
    args=parser.parse_args()
    
    ''' default all args are strings'''
    instance_file=args.instance
    algo=args.algorithm
    randomseed=int(args.randomSeed)
    if args.epsilon:
        epsilon=float(args.epsilon)
    horizon=int(args.horizon)

    '''get the list of true means'''
    fh = open(instance_file)
    lines = fh.readlines()
    list_bandit_instnce=[float(line) for line in lines]
    fh.close()
    np.random.seed(randomseed)
    '''parse acoording to algo'''
    if(algo=="epsilon-greedy"):
        g=epsilonGreedy(epsilon,[],[],list_bandit_instnce)
        g.preProcess(len(list_bandit_instnce))
        g.run(horizon)
        REG=g.getRegret()
        print("{}, {}, {}, {}, {}, {}".format(instance_file,algo,randomseed,epsilon,horizon,REG))
    if(algo=="ucb"):
        g=Ucb(epsilon,[],[],list_bandit_instnce)
        g.preProcess(len(list_bandit_instnce))
        g.run(horizon)
        REG=g.getRegret()
        print("{}, {}, {}, {}, {}, {}".format(instance_file,algo,randomseed,epsilon,horizon,REG))

    if(algo=="kl-ucb"):
        g=KLUCB(epsilon,[],[],list_bandit_instnce)
        g.preProcess(len(list_bandit_instnce))
        g.run(horizon)
        REG=g.getRegret()
        print("{}, {}, {}, {}, {}, {}".format(instance_file,algo,randomseed,epsilon,horizon,REG))
    
    if(algo=="thompson-sampling"):
        g=thompsonSampling(epsilon,[],[],[],[],list_bandit_instnce)
        g.preProcess(len(list_bandit_instnce))
        g.run(horizon)
        REG=g.getRegret()
        print("{}, {}, {}, {}, {}, {}".format(instance_file,algo,randomseed,epsilon,horizon,REG))

    if(algo=="thompson-sampling-with-hint"):
        g=thompsonSamplinghint(epsilon,[],[],[],[],list_bandit_instnce,np.sort(list_bandit_instnce)[::-1])
        g.preProcess(len(list_bandit_instnce))
        g.run(horizon)
        REG=g.getRegret()
        print("{}, {}, {}, {}, {}, {}".format(instance_file,algo,randomseed,epsilon,horizon,REG))

    # print(t-time.time())