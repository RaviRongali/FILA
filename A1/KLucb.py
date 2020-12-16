import numpy as np
class KLUCB():
    def __init__(self,ep,b_pulls,b_means,true_means):
        self.ep=ep
        self.b_pulls=b_pulls
        # self.b_rewards=b_rewards
        self.b_means=b_means
        self.true_means=true_means
        self.cumRew=0
        self.horizon=0
    #initialse
    def preProcess(self,nof_arms):
        self.b_pulls=[0 for i in range(nof_arms)]
        self.b_rewards=[0.0 for i in range(nof_arms)]
        self.b_means=[0.0 for i in range(nof_arms)]

    def KL(self,a,b):
        # print("a={} b={}".format(a,b))
        if a==0:
            return np.log(1/(1-b))
        if a==b:
            return 0
        return a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))
    def get_max_q(self,prec,c,pa,t,ua):
        bmin=pa
        bmax=1
        ans=(float(pa)+1)/2
        lim=np.log(t)+c*np.log(np.log(t))
        # print("limit= ",lim)
        while(1):
            z=ua*self.KL(float(pa),float(ans))
            # print("closing q= ",ans,z)
            if(z<=lim):
                if(lim-z<prec):
                    return ans
                bmin=ans
                ans=(float(ans)+float(bmax))/2
            else:
                bmax=ans
                ans=(float(ans)+float(bmin))/2
                
    #pull_arm based on ucb value list,pull each arm ones
    def pullArm(self,precision=1e-3,c=3):
        arms=len(self.b_pulls)
        for arm in range(arms):
            if self.b_pulls[arm]==0:
                return arm
        kl_ucb_arms=[0.0 for i in range(arms)]
        t=sum(self.b_pulls)+1
        for arm in range(arms):
            ut=self.b_pulls[arm]
            pa=self.b_means[arm]
            if pa==1.0:
                kl_ucb_arms[arm]=pa
            else:
                max_q=self.get_max_q(precision,c,pa,t,ut)
                # print("arm={} found q={} where bmean={}".format(arm,max_q,self.b_means[arm]))
                kl_ucb_arms[arm]=max_q
        return np.argmax(kl_ucb_arms)
    #update reward
    def updateExpmean(self,arm,reward):
        self.b_pulls[arm]=self.b_pulls[arm]+1
        # self.b_rewards[arm]=self.b_rewards[arm]+1
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