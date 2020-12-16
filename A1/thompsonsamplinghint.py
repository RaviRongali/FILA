import numpy as np
class thompsonSamplinghint():
    def __init__(self,ep,b_pulls,s_pulls,f_pulls,b_means,true_means,given_means):
        self.ep=ep
        self.b_pulls=b_pulls
        self.s_pulls=s_pulls
        self.f_pulls=f_pulls
        self.b_means=b_means
        self.true_means=true_means
        self.given_means=given_means
        self.belifs=[]
        self.cumRew=0
        self.horizon=0
        # print(self.true_means)
        # print(self.given_means)
    #initialse
    def preProcess(self,nof_arms):
        self.b_pulls=[0 for i in range(nof_arms)]
        self.s_pulls=[0 for i in range(nof_arms)]
        self.f_pulls=[0 for i in range(nof_arms)]
        self.b_rewards=[0.0 for i in range(nof_arms)]
        self.b_means=[0.0 for i in range(nof_arms)]
        befliefs=[1/float(nof_arms)]*nof_arms
        # befliefs=[1/float(nof_arms) for i in range(nof_arms)]
        # for i in range(nof_arms):
        #     self.belifs.append(befliefs)
        for i in range(nof_arms):
            self.belifs.append([1/float(nof_arms)]*nof_arms)
        # print(self.belifs)
                
    #pull_arm based on ucb value list,pull each arm ones
    def pullArm(self):
        arms=len(self.b_pulls)
        max_means=[0.0 for i in range(arms)]
        for arm in range(arms):
            # print(self.belifs[arm])
            # max_means[arm]=np.random.choice(self.given_means,1,p=self.belifs[arm])[0]
            max_means[arm]=self.belifs[arm][0]
        # print(max_means)
        return np.argmax(max_means)
    #update reward
    def updateExpmean(self,arm,reward):
        self.b_pulls[arm]=self.b_pulls[arm]+1
        arms=len(self.b_means)
        if reward==1:
            # self.s_pulls[arm]=self.s_pulls[arm]+1
            deno=0
            for i in range(arms):
                deno=deno+self.given_means[i]*self.belifs[arm][i]
            for i in range(arms):
                self.belifs[arm][i]=self.belifs[arm][i]*self.given_means[i]/deno
        else:
            # self.f_pulls[arm]=self.f_pulls[arm]+1
            deno=0
            for i in range(arms):
                deno=deno+(1-self.given_means[i])*self.belifs[arm][i]
            for i in range(arms):
                self.belifs[arm][i]=self.belifs[arm][i]*(1-self.given_means[i])/deno
        # print(self.belifs)
    
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
            # break
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
        # print(self.belifs)
        return regret