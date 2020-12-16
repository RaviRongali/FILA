import numpy as np
import matplotlib.pyplot as plt
class TDsarsa:
    def __init__(self,states,start,end,actions,alpha,epsilon,gamma,episodes,wind,stocasticity):
        self.states=states
        self.start=start
        self.end=end
        self.alpha=alpha
        self.actions=actions
        self.gamma=gamma
        self.episodes=episodes
        self.epsilon=epsilon
        self.wind=wind
        self.timescount=np.zeros(self.episodes)
        self.stocasticity=stocasticity

    def epsionGreedy(self,b):
        p=np.random.random()
        if p<self.epsilon:
            '''explore'''
            return np.random.randint(len(b))
        else:
            '''exploit'''
            return np.argmax(b)

    def transition(self,curState,action):
        '''def 0,1,2,3 as E,N,W,S anti clockwise as 4,5,6,7 NE,NW,SW,SE'''
        nextState=curState
        if action==0:
            nextState=(nextState[0],nextState[1]+1)
        elif action==1:
            nextState=(nextState[0]-1,nextState[1])
        elif action==2:
            nextState=(nextState[0],nextState[1]-1)
        elif action==3:
            nextState=(nextState[0]+1,nextState[1])
        elif action==4:
            nextState=(nextState[0]-1,nextState[1]+1)
        elif action==5:
            nextState=(nextState[0]-1,nextState[1]-1)
        elif action==6:
            nextState=(nextState[0]+1,nextState[1]-1)
        elif action==7:
            nextState=(nextState[0]+1,nextState[1]+1)
        elif action==8:
            nextState=nextState

        # if nextState[1]>self.states[1]-1:
        #     nextState=(nextState[0],self.states[1]-1)
        # if nextState[1]<0:
        #     nextState=(nextState[0],0)


        '''add wind'''
        '''if stochasity is 1 add -1,0,1'''
        k=0
        if self.stocasticity==1:
            k=np.random.choice([-1,0,1])
        nextState=(nextState[0]-self.wind[curState[1]]-k,nextState[1])

        if nextState[0]>self.states[0]-1:
            nextState=(self.states[0]-1,nextState[1])
        if nextState[1]>self.states[1]-1:
            nextState=(nextState[0],self.states[1]-1)
        if nextState[1]<0:
            nextState=(nextState[0],0)
        if nextState[0]<0:
            nextState=(0,nextState[1])

        return nextState

    def Sarsa(self):
        Q=np.zeros((self.states[0],self.states[1],self.actions))
        steps=0
        for i in range(self.episodes):
            curState=self.start
            action=self.epsionGreedy(Q[curState[0]][curState[1]])
            while(curState!=self.end):
                reward=-1
                nextstate=self.transition(curState,action)
                if nextstate==self.end:
                    reward=1
                nextaction=self.epsionGreedy(Q[nextstate[0]][nextstate[1]])
                target=reward+self.gamma*Q[nextstate[0]][nextstate[1]][nextaction]
                Q[curState[0]][curState[1]][action]=Q[curState[0]][curState[1]][action]+self.alpha*(target-Q[curState[0]][curState[1]][action])
                curState=nextstate
                action=nextaction
                steps=steps+1
            self.timescount[i]=steps
        # plt.figure(1)
        # x=[i for i in range(self.episodes)]
        # print(len(x))
        # print(len(self.timescount))
        # plt.plot(x,self.timescount,label="test")
        # plt.savefig('part_c')
        return self.timescount

    def ExpectedSarsa(self):
        Q=np.zeros((self.states[0],self.states[1],self.actions))
        steps=0
        for i in range(self.episodes):
            curState=self.start
            action=self.epsionGreedy(Q[curState[0]][curState[1]])
            while(curState!=self.end):
                reward=-1
                nextstate=self.transition(curState,action)
                if nextstate==self.end:
                    reward=1
                nextaction=self.epsionGreedy(Q[nextstate[0]][nextstate[1]])
                target=0
                # self.gamma*Q[nextstate[0]][nextstate[1]][nextaction]
                maxQa=np.argmax(Q[nextstate[0]][nextstate[1]])
                # print(maxQa,"111")
                exp=self.epsilon/self.actions
                for j in range(self.actions):
                    if j==maxQa:
                        target=target+(1-self.epsilon+exp)*Q[nextstate[0]][nextstate[1]][j]
                    else:
                        target=target+exp*Q[nextstate[0]][nextstate[1]][j]
                target=reward+self.gamma*(target)
                Q[curState[0]][curState[1]][action]=Q[curState[0]][curState[1]][action]+self.alpha*(target-Q[curState[0]][curState[1]][action])
                curState=nextstate
                action=nextaction
                steps=steps+1
                # print(steps)
            self.timescount[i]=steps
            # print(steps)
        return self.timescount

    def Qlearning(self):
        Q=np.zeros((self.states[0],self.states[1],self.actions))
        steps=0
        for i in range(self.episodes):
            curState=self.start
            action=self.epsionGreedy(Q[curState[0]][curState[1]])
            while(curState!=self.end):
                reward=-1
                nextstate=self.transition(curState,action)
                if nextstate==self.end:
                    reward=1
                nextaction=self.epsionGreedy(Q[nextstate[0]][nextstate[1]])
                target=reward+self.gamma*(max(Q[nextstate[0]][nextstate[1]]))
                Q[curState[0]][curState[1]][action]=Q[curState[0]][curState[1]][action]+self.alpha*(target-Q[curState[0]][curState[1]][action])
                curState=nextstate
                action=nextaction
                steps=steps+1
            self.timescount[i]=steps
        return self.timescount