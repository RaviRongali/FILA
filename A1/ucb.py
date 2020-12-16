import numpy as np
class Ucb():
    def __init__(self,ep,b_pulls,b_means,true_means):
        self.ep=ep
        self.b_pulls=b_pulls
        self.b_means=b_means
        self.true_means=true_means
        self.cumRew=0
        self.horizon=0
    #initialse
    def preProcess(self,nof_arms):
        self.b_pulls=[0 for i in range(nof_arms)]
        self.b_means=[0.0 for i in range(nof_arms)]
    #pull_arm based on ucb value list,pull each arm ones
    def pullArm(self):
        arms=len(self.b_pulls)
        for arm in range(arms):
            if self.b_pulls[arm]==0:
                return arm
        ucb_arms=[0.0 for i in range(arms)]
        t=sum(self.b_pulls)
        for arm in range(arms):
            ut=self.b_pulls[arm]
            ucb_val=np.sqrt((2*np.log(t))/float(ut))
            ucb_arms[arm]=self.b_means[arm]+ucb_val
        return np.argmax(ucb_arms)
    #update reward
    def updateExpmean(self,arm,reward):
        self.b_pulls[arm]=self.b_pulls[arm]+1
        n=self.b_pulls[arm]
        mean=self.b_means[arm]
        new_mean=((n-1)*float(mean)+reward)/float(n)
        self.b_means[arm]=new_mean
    
    def getReward(self,arm):
        p=np.random.random()
        if p>self.true_means[arm]:
            return 0.0
        else:
            return 1.0
    #run hz time
    def run(self,horizon):
        self.horizon=horizon
        for i in range(horizon):
            arm=self.pullArm()
            reward=self.getReward(arm)
            self.cumRew=self.cumRew+reward
            self.updateExpmean(arm,reward)
            # print(self.b_means)

    def getRegret(self):
        max_true_mean=max(self.true_means)
        regret=max_true_mean*self.horizon-self.cumRew
        # print(self.true_means)
        # print(self.b_means)
        # print("max =",max_true_mean*self.horizon," cum reward =",self.cumRew)
        return regret