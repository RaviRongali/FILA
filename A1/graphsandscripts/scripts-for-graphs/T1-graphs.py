import matplotlib
import matplotlib.pyplot as plt
import numpy as np


instances=["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
horizons=[100, 400, 1600, 6400, 25600, 102400]
seeds=[i for i in range(50)]

greedy=[0 for i in range(len(horizons))]
uc=[0 for i in range(len(horizons))]
kl_uc=[0 for i in range(len(horizons))]
thompson=[0 for i in range(len(horizons))]

greedy1=[0 for i in range(len(horizons))]
uc1=[0 for i in range(len(horizons))]
kl_uc1=[0 for i in range(len(horizons))]
thompson1=[0 for i in range(len(horizons))]

greedy2=[0 for i in range(len(horizons))]
uc2=[0 for i in range(len(horizons))]
kl_uc2=[0 for i in range(len(horizons))]
thompson2=[0 for i in range(len(horizons))]
f=open("outputDataT1.txt")
lines=f.readlines()
for line in lines:
    line=line.rstrip()
    line=line.split(",")
    inst=line[0].strip()
    al=line[1].strip()
    if inst=="../instances/i-1.txt" and al=="epsilon-greedy":
        hr=int(line[4])
        index=horizons.index(hr)
        greedy[index]=greedy[index]+float(line[5])
    if inst=="../instances/i-2.txt" and al=="epsilon-greedy":
        hr=int(line[4])
        index=horizons.index(hr)
        greedy1[index]=greedy1[index]+float(line[5])
    if inst=="../instances/i-3.txt" and al=="epsilon-greedy":
        hr=int(line[4])
        index=horizons.index(hr)
        greedy2[index]=greedy2[index]+float(line[5])
    if inst=="../instances/i-1.txt" and al=="ucb":
        hr=int(line[4])
        index=horizons.index(hr)
        uc[index]=uc[index]+float(line[5])
    if inst=="../instances/i-2.txt" and al=="ucb":
        hr=int(line[4])
        index=horizons.index(hr)
        uc1[index]=uc1[index]+float(line[5])
    if inst=="../instances/i-3.txt" and al=="ucb":
        hr=int(line[4])
        index=horizons.index(hr)
        uc2[index]=uc2[index]+float(line[5])

    if inst=="../instances/i-1.txt" and al=="kl-ucb":
        hr=int(line[4])
        index=horizons.index(hr)
        kl_uc[index]=kl_uc[index]+float(line[5])
    if inst=="../instances/i-2.txt" and al=="kl-ucb":
        hr=int(line[4])
        index=horizons.index(hr)
        kl_uc1[index]=kl_uc1[index]+float(line[5])
    if inst=="../instances/i-3.txt" and al=="kl-ucb":
        hr=int(line[4])
        index=horizons.index(hr)
        kl_uc2[index]=kl_uc2[index]+float(line[5])

    if inst=="../instances/i-1.txt" and al=="thompson-sampling":
        hr=int(line[4])
        index=horizons.index(hr)
        thompson[index]=thompson[index]+float(line[5])
    if inst=="../instances/i-2.txt" and al=="thompson-sampling":
        hr=int(line[4])
        index=horizons.index(hr)
        thompson1[index]=thompson1[index]+float(line[5])
    if inst=="../instances/i-3.txt" and al=="thompson-sampling":
        hr=int(line[4])
        index=horizons.index(hr)
        thompson2[index]=thompson2[index]+float(line[5])
f.close()

greedy=[i/50 for i in greedy]
uc=[i/50 for i in uc]
kl_uc=[i/50 for i in kl_uc]
thompson=[i/50 for i in thompson]

greedy1=[i/50 for i in greedy1]
uc1=[i/50 for i in uc1]
kl_uc1=[i/50 for i in kl_uc1]
thompson1=[i/50 for i in thompson1]

greedy2=[i/50 for i in greedy2]
uc2=[i/50 for i in uc2]
kl_uc2=[i/50 for i in kl_uc2]
thompson2=[i/50 for i in thompson2]

# print(greedy,greedy1,greedy2,uc,uc1,uc2,kl_uc,kl_uc1,kl_uc2,thompson,thompson1,thompson2)

'''for instance-1'''
def dataplot(i,g,u,k,t,title,safeimg):
    plt.figure(i)
    t = [np.log(i) for i in horizons]
    plt.plot(t, g, label = "greedy")
    plt.plot(t, u, label = "ucb")
    plt.plot(t, k, label = "kl-ucb")
    plt.plot(t, t, label = "thompson-sampling")
    plt.xlabel('Horizons')
    plt.ylabel('Regret')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig(safeimg)

dataplot(1,greedy,uc,kl_uc,thompson,"for instances/i-1.txt","T1instance-1.png")
dataplot(2,greedy1,uc1,kl_uc1,thompson1,"for instances/i-2.txt","T1instance-2.png")
dataplot(3,greedy2,uc2,kl_uc2,thompson2,"for instances/i-3.txt","T1instance-3.png")


