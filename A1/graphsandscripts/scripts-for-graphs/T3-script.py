import os
import subprocess

instances=["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
epsilon=[0.001,0.01,0.1]
horizons=[102400]
seeds=[i for i in range(50)]

regret=[0.0,0.0,0.0]
regret1=[0.0,0.0,0.0]
regret2=[0.0,0.0,0.0]
# f = open("outputDataT3.txt", "w")
for inst in instances:
    for epsi in epsilon:
        for sd in seeds:
            # if hr<2 and sd<3:
            cmd="python3 bandit.py --instance "+ inst+ " --algorithm epsilon-greedy"+ " --randomSeed " +str(sd)+ " --epsilon " +str(epsi)+ " --horizon " +str(102400)
            k = subprocess.check_output(cmd, shell=True).decode("utf-8").rstrip()
            line=k.split(",")
            inst=line[0].strip()
            ep=float(line[3].strip())
            print(k)
            if inst=="../instances/i-1.txt" and ep==epsilon[0]:
                regret[0]=regret[0]+float(line[5])
            if inst=="../instances/i-1.txt" and ep==epsilon[1]:
                regret[1]=regret[1]+float(line[5])
            if inst=="../instances/i-1.txt" and ep==epsilon[2]:
                regret[2]=regret[2]+float(line[5])
            if inst=="../instances/i-2.txt" and ep==epsilon[0]:
                regret1[0]=regret1[0]+float(line[5])
            if inst=="../instances/i-2.txt" and ep==epsilon[1]:
                regret1[1]=regret1[1]+float(line[5])
            if inst=="../instances/i-2.txt" and ep==epsilon[2]:
                regret1[2]=regret1[2]+float(line[5])
            if inst=="../instances/i-3.txt" and ep==epsilon[0]:
                regret2[0]=regret2[0]+float(line[5])
            if inst=="../instances/i-3.txt" and ep==epsilon[1]:
                regret2[1]=regret2[1]+float(line[5])
            if inst=="../instances/i-3.txt" and ep==epsilon[2]:
                regret2[2]=regret2[2]+float(line[5])
            # f.write(k)
            # f.write("\n")
# f.close()   
regret=[i/float(len(seeds)) for i in regret]
regret1=[i/float(len(seeds)) for i in regret1]
regret2=[i/float(len(seeds)) for i in regret2]
print( "instance-1 regret[0.001,0.01,0.1]={}\n".format(regret))
print( "instance-2 regret[0.001,0.01,0.1]={}\n".format(regret1))
print( "instance-3 regret[0.001,0.01,0.1]={}\n".format(regret2))