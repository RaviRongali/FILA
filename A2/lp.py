import numpy as np
from pulp import *
class linearProgramming():
    def __init__(self,nS,nA,transitions,discount):
        self.nS=nS
        self.nA=nA
        self.transitions=transitions
        self.discount=discount
        self.v=np.zeros(nS)
        self.optimalA=np.zeros(self.nS)

    def Q(self,s,v):
        A=np.zeros(self.nA)
        for a in range(self.nA):
            for b in self.transitions[s][a]:
                nextS=b[0]
                TsaS=b[2]
                RsaS=b[1]
                A[a]=A[a]+(TsaS*(RsaS+self.discount*v[nextS]))
        return A
    
    def V_of_lp(self):
        
        vars = LpVariable.dicts("v", range(self.nS)) 
        prob = LpProblem("lp", LpMinimize)
        # print(var_state)
        prob += lpSum([vars[i] for i in range(self.nS)])
        # adding nk constaints and solving
        for eachState in range(self.nS):
            for action in range(self.nA):
                vf=0
                for b in self.transitions[eachState][action]:
                    nextS=b[0]
                    TsaS=b[2]
                    RsaS=b[1]
                    vf=vf+(TsaS*(RsaS+self.discount*vars[nextS]))
                # adding this states paticular action constraint
                prob +=vars[eachState]>=vf
        LpSolverDefault.msg = 0 
        prob.solve()
        
        Vpi=np.zeros(self.nS)
        for v in prob.variables():
            # print(v.name.split("_")[1])
            k=v.name.split("_")[1]
            Vpi[int(k)]=v.varValue
        
        # Vpi=np.zeros(self.nS)
        return Vpi


    def run(self):
        vpi=self.V_of_lp()
        for eachstate in range(self.nS):
            As=self.Q(eachstate,vpi)
            optaction=np.argmax(As)
            self.optimalA[eachstate]=optaction
            self.v[eachstate]=vpi[eachstate]
    def lpprint(self):
        for i in range(self.nS):
            print("{}\t{}".format(self.v[i],int(self.optimalA[i])))






