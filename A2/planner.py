import argparse
import numpy as np
import time
from VI import valueIteration
from hpi import howardspolicyIteration
from lp import linearProgramming
if __name__=="__main__":
    t=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument("--mdp",help="mdp file")
    parser.add_argument("--algorithm",help="algorithm vi,hpi or li")
    args=parser.parse_args()

    ''' default all args are strings'''
    mdpFile=args.mdp
    algo=args.algorithm
    np.random.seed(0)
    ''' Parsing '''
    fh = open(mdpFile)
    lines = fh.readlines()
    nolines=len(lines)
    states=int(lines[0].rstrip().split(" ")[1])
    actions=int(lines[1].rstrip().split(" ")[1])
    discount=float(lines[nolines-1].rstrip().split(" ")[-1])
    mdptype=lines[nolines-2].rstrip().split(" ")[-1]
    transitions=[]
    for i in range(states):
        trans=[]
        for j in range(actions):
            trans.append([])
        transitions.append(trans)
    for i in range(4,nolines-2):
        trans=lines[i].rstrip().split(" ")
        transitions[int(trans[1])][int(trans[2])].append([int(trans[3]),float(trans[4]),float(trans[5])])
    # print(transitions)
    # print(states,actions,discount,mdptype)
    fh.close()

    if(algo=="vi"):
        vi=valueIteration(states,actions,transitions,discount)
        vi.run()
        vi.Viprint()

    if(algo=="hpi"):
        # print(transitions,states,actions,discount)
        hpit=howardspolicyIteration(states,actions,transitions,discount)
        # hpit.VS()
        hpit.run()
        hpit.hpiprint()

    if(algo=="lp"):
        Lp=linearProgramming(states,actions,transitions,discount)
        Lp.run()
        Lp.lpprint()
        pass


