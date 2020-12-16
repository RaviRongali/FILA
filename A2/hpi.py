import numpy as np
class howardspolicyIteration():
    def __init__(self,nS,nA,transitions,discount):
        self.nS=nS
        self.nA=nA
        self.transitions=transitions
        self.discount=discount
        self.v=np.zeros(nS)
        self.optimalA=[]
        for i in range(self.nS):
            p=[]
            for a in range(len(self.transitions[i])):
                if len(self.transitions[i][a])!=0:
                    p.append(a)
            if (len(p))!=0:
                self.optimalA.append(p[0])
            else:
                self.optimalA.append(0)
        # print("randoptimal",self.optimalA)
        # for i in range(self.nS):
        #     print(transitions[i])
    def VS(self,pi):
        A=[]
        B=[0]*self.nS
        for i in range(self.nS):
            A.append([0]*self.nS)
        for i in range(self.nS):
            action=pi[i]
            A[i][i]=1
            c=0
            for b in self.transitions[i][action]:
                nextS=b[0]
                TsaS=b[2]
                RsaS=b[1]
                nextScoeff=TsaS*self.discount
                A[i][nextS]=A[i][nextS]-1*nextScoeff
                c=c+TsaS*RsaS
            B[i]=c
        #print(A,B)
        # for a in range(len(A)):
        #     print(A[a])
        # print(B)
        x = np.linalg.solve(A, B)
        # x=np.linalg.pinv(A,B)
        # print(x)
        return x
        
    def Q(self,s,v):
        A=np.zeros(self.nA)
        for a in range(self.nA):
            for b in self.transitions[s][a]:
                nextS=b[0]
                TsaS=b[2]
                RsaS=b[1]
                A[a]=A[a]+(TsaS*(RsaS+self.discount*v[nextS]))
        return A
    
    def run(self):
        imp=self.optimalA
        # print(imp)
        c=0
        while(1):
            c=c+1
            # print(c)
            Vphi=self.VS(imp)
            # print(Vphi)
            count=0
            for eachState in range(self.nS):
                A=self.Q(eachState,Vphi)
                action=self.optimalA[eachState]
                qval=A[action]
                for ac in range(self.nA):
                    if A[ac]>qval:
                        action=ac
                        break
                if(action==imp[eachState]):
                    count=count+1
                else:
                    imp[eachState]=action
                    self.optimalA[eachState]=action
            if count==self.nS:
                self.v=Vphi
                break

    def hpiprint(self):
        for i in range(self.nS):
            print("{}\t{}".format(self.v[i],int(self.optimalA[i])))






