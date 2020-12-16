import numpy as np
class thompsonSampling():
    def __init__(self,ep,b_pulls,s_pulls,f_pulls,b_means,true_means):
        self.ep=ep
        self.b_pulls=b_pulls
        self.s_pulls=s_pulls
        self.f_pulls=f_pulls
        self.b_means=b_means
        self.true_means=true_means
        self.cumRew=0
        self.horizon=0
    #initialse
    def preProcess(self,nof_arms):
        self.b_pulls=[0 for i in range(nof_arms)]
        self.s_pulls=[0 for i in range(nof_arms)]
        self.f_pulls=[0 for i in range(nof_arms)]
        self.b_rewards=[0.0 for i in range(nof_arms)]
        self.b_means=[0.0 for i in range(nof_arms)]
                
    #pull_arm based on ucb value list,pull each arm ones
    def pullArm(self,precision=1e-3,c=3):
        arms=len(self.b_pulls)
        # for arm in range(arms):
        #     if self.b_means[arm]==0:
        #         return arm
        beta_random=[0.0 for i in range(arms)]
        for arm in range(arms):
            sa=self.s_pulls[arm]
            fa=self.f_pulls[arm]
            beta_random[arm]=np.random.beta(float(sa)+1,float(fa)+1)
        # print(beta_random)
        return np.argmax(beta_random)
    #update reward
    def updateExpmean(self,arm,reward):
        self.b_pulls[arm]=self.b_pulls[arm]+1
        if reward==1:
            self.s_pulls[arm]=self.s_pulls[arm]+1
        else:
            self.f_pulls[arm]=self.f_pulls[arm]+1
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