import argparse
import numpy as np
import time
from TDSarsa import TDsarsa 
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--grid",help="grid file")
    parser.add_argument("--wind",help="wind file")
    parser.add_argument("--actions",help="no of actions")
    parser.add_argument("--episodes",help="episodes")
    parser.add_argument("--task",help="taskno")
    # parser.add_argument("--gamma",help="gamma")
    # parser.add_argument("--epsilon",help="epsilon")
    # parser.add_argument("--alpha",help="alpha")
    
    parser.add_argument("--withstocasticity",help="1 if with stocasticity")
    args=parser.parse_args()
    '''constants or inputs'''
    stocasticity=0
    episodes=170
    gamma=1
    alpha=0.5
    epsilon=0.1
    ''' default all args are strings'''
    gridFile=args.grid
    windFile=args.wind
    task=int(args.task)
    actions=int(args.actions)
    if args.withstocasticity:
        stocasticity=int(args.withstocasticity)
    ''' Parsing '''
    fg = open(gridFile)
    lines = fg.readlines()
    grid = np.array([line.split() for line in lines], dtype=np.int64)
    l,b = grid.shape
    fg.close()

    fw = open(windFile)
    lines = fw.readlines()
    wind=np.array([line.split() for line in lines], dtype=np.int64)[0]
    fw.close()

    start=(0,0)
    end=(0,0)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j]==1:
                start=(i,j)
            if grid[i][j]==2:
                end=(i,j)

    # print("start",start)
    # print("end",end)
    # print("wind",wind)

    '''
        input size of states,actions,episodes,gamma,epsilon
    '''

    i
    states=(l,b)
    if task!=4 and task!=5:
        sarsaT=np.zeros(episodes)
        for i in range(10):
            np.random.seed(i)
            algo=TDsarsa(states,start,end,actions,alpha,epsilon,gamma,episodes,wind,stocasticity)
            t=algo.Sarsa()
            sarsaT=[a+b for a,b in zip(t,sarsaT)]
        '''plot'''
        sarsaT=[i/10 for i in sarsaT]
        sarsaT.insert(0,0)
        sarsaT=np.ceil(sarsaT)
        print(sarsaT)
        
        plt.figure(1)
        x=[i for i in range(0,episodes+1)]
        # for i in range(1,episodes+1):
        #     print("episode={},timesteps={}".format(i,sarsaT[i]))
        plt.plot(sarsaT,x,label="test")
        plotname="Task_1"
        title="four actions non stochastic sarsa"
        # if task==1:
        #     plt.yticks([0,50,100,150,170,175])
        if task==2:
            plotname="Task_2"
            title="eight actions non stochastic sarsa"
        if task==3:
            plotname="Task_3"
            title="eight actions  stochastic sarsa"

        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.title(title)
        plt.savefig(plotname)

    if task==4:
        sarsaT=np.zeros(episodes)
        ExpsarsaT=np.zeros(episodes)
        QlearningT=np.zeros(episodes)
        for i in range(10):
            np.random.seed(i)
            algo=TDsarsa(states,start,end,actions,alpha,epsilon,gamma,episodes,wind,stocasticity)
            t=algo.Sarsa()
            sarsaT=[a+b for a,b in zip(t,sarsaT)]

        for i in range(10):
            np.random.seed(i)
            algo=TDsarsa(states,start,end,actions,alpha,epsilon,gamma,episodes,wind,stocasticity)
            t=algo.ExpectedSarsa()
            ExpsarsaT=[a+b for a,b in zip(t,ExpsarsaT)]
        
        for i in range(10):
            np.random.seed(i)
            algo=TDsarsa(states,start,end,actions,alpha,epsilon,gamma,episodes,wind,stocasticity)
            t=algo.Qlearning()
            QlearningT=[a+b for a,b in zip(t,QlearningT)]
        
        sarsaT=[i/10 for i in sarsaT]
        sarsaT.insert(0,0)
        sarsaT=np.ceil(sarsaT)
        print(sarsaT)

        ExpsarsaT=[i/10 for i in ExpsarsaT]
        ExpsarsaT.insert(0,0)
        ExpsarsaT=np.ceil(ExpsarsaT)
        print(ExpsarsaT)
        

        QlearningT=[i/10 for i in QlearningT]
        QlearningT.insert(0,0)
        QlearningT=np.ceil(QlearningT)
        print(QlearningT)
        

        plt.figure(1)
        x=[i for i in range(0,episodes+1)]
        plt.plot(sarsaT,x,label="sarsa")
        plt.plot(ExpsarsaT,x,label="expectedsarsa")
        plt.plot(QlearningT,x,label="qlearning")
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.title('Comparing 3 control alogrithms for 4 actions')
        plt.legend(loc='upper left')
        plt.savefig('Task_4')
    '''9 states'''

    if task==5:
        sarsaT=np.zeros(episodes)
        ExpsarsaT=np.zeros(episodes)
        # QlearningT=np.zeros(episodes)
        for i in range(10):
            np.random.seed(i)
            algo=TDsarsa(states,start,end,actions-1,alpha,epsilon,gamma,episodes,wind,stocasticity)
            t=algo.Sarsa()
            sarsaT=[a+b for a,b in zip(t,sarsaT)]
        for i in range(10):
            np.random.seed(i)
            algo=TDsarsa(states,start,end,actions,alpha,epsilon,gamma,episodes,wind,stocasticity)
            t=algo.Sarsa()
            ExpsarsaT=[a+b for a,b in zip(t,ExpsarsaT)]
        
        # for i in range(10):
        #     np.random.seed(i)
        #     algo=TDsarsa(states,start,end,actions,alpha,epsilon,gamma,episodes,wind,stocasticity)
        #     t=algo.Qlearning()
        #     QlearningT=[a+b for a,b in zip(t,QlearningT)]
        
        sarsaT=[i/10 for i in sarsaT]
        sarsaT.insert(0,0)
        sarsaT=np.ceil(sarsaT)
        print(sarsaT)

        ExpsarsaT=[i/10 for i in ExpsarsaT]
        ExpsarsaT.insert(0,0)
        ExpsarsaT=np.ceil(ExpsarsaT)
        print(ExpsarsaT)
        

        # QlearningT=[i/10 for i in QlearningT]
        # QlearningT.insert(0,0)
        # QlearningT=np.ceil(QlearningT)
        # print(QlearningT)
        

        plt.figure(1)
        x=[i for i in range(0,episodes+1)]
        plt.plot(sarsaT,x,label="sarsa with 8")
        plt.plot(ExpsarsaT,x,label="sarsa with 9")
        # plt.plot(QlearningT,x,label="qlearning")
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.title('Comparing with one more action 9 with stocasticity')
        plt.legend(loc='upper left')
        plt.savefig('Task_5')




    # if(algo=="vi"):
    #     vi=valueIteration(states,actions,transitions,discount)
    #     vi.run()
    #     vi.Viprint()



