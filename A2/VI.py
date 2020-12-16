import numpy as np
class valueIteration():
    def __init__(self,nS,nA,transitions,discount):
        self.nS=nS
        self.nA=nA
        self.transitions=transitions
        self.discount=discount
        self.v=np.zeros(nS)
        self.optimalA=np.zeros(nS)
    
    def Q(self,s,v):
        A=np.zeros(self.nA)
        for a in range(self.nA):
            # print("given a= ",a,self.transitions[s][a])
            for b in self.transitions[s][a]:
                nextS=b[0]
                TsaS=b[2]
                RsaS=b[1]
                A[a]=A[a]+(TsaS*(RsaS+self.discount*v[nextS]))
        return A

    def Vf(self,s):
        qsa=self.Q(s,self.v)
        self.optimalA[s]=np.argmax(qsa)
        return np.max(qsa)

    
    def run(self,precision=0.00000001):
        check=np.zeros(self.nS)
        count=0
        while(1):
            count=count+1
            # print(count)
            for eachState in range(self.nS):
                curr=self.Vf(eachState)
                check[eachState]=np.abs(curr-self.v[eachState])
                # difference=max(difference,diff)
                # print(curr,self.v[eachState],diff)
                self.v[eachState]=curr
            maxi=0
            for eachState in range(self.nS):
                maxi=max(maxi,check[eachState])
            if maxi<precision:
                # print(check)
                break



    def Viprint(self):
        for i in range(self.nS):
            print("{}\t{}".format(self.v[i],int(self.optimalA[i])))






