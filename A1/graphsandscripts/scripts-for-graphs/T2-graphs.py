import matplotlib
import matplotlib.pyplot as plt
import numpy as np


instances=["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
horizons=[100, 400, 1600, 6400, 25600, 102400]
seeds=[i for i in range(50)]


thompson=[0 for i in range(len(horizons))]
thompson_hint=[0 for i in range(len(horizons))]

thompson1=[0 for i in range(len(horizons))]
thompson_hint1=[0 for i in range(len(horizons))]

thompson2=[0 for i in range(len(horizons))]
thompson_hint2=[0 for i in range(len(horizons))]

f=open("outputDataT2.txt")
lines=f.readlines()
for line in lines:
    line=line.rstrip()
    line=line.split(",")
    inst=line[0].strip()
    al=line[1].strip()
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

    if inst=="../instances/i-1.txt" and al=="thompson-sampling-with-hint":
        hr=int(line[4])
        index=horizons.index(hr)
        thompson_hint[index]=thompson_hint[index]+float(line[5])
    if inst=="../instances/i-2.txt" and al=="thompson-sampling-with-hint":
        hr=int(line[4])
        index=horizons.index(hr)
        thompson_hint1[index]=thompson_hint1[index]+float(line[5])
    if inst=="../instances/i-3.txt" and al=="thompson-sampling-with-hint":
        hr=int(line[4])
        index=horizons.index(hr)
        thompson_hint2[index]=thompson_hint2[index]+float(line[5])
f.close()


thompson=[i/50 for i in thompson]
print("\nthompson i-1.txt",thompson)

thompson_hint=[i/50 for i in thompson_hint]
print("\nthompson hint i-1.txt",thompson_hint)

thompson1=[i/50 for i in thompson1]
print("\nthompson i-2.txt",thompson1)

thompson_hint1=[i/50 for i in thompson_hint1]
print("\nthompson hint i-2.txt",thompson_hint1)

thompson2=[i/50 for i in thompson2]
print("\nthompson i-3.txt",thompson2)

thompson_hint2=[i/50 for i in thompson_hint2]
print("\nthompson hint i-3.txt",thompson_hint2)




# print(greedy,greedy1,greedy2,uc,uc1,uc2,kl_uc,kl_uc1,kl_uc2,thompson,thompson1,thompson2)

'''for instance-1'''
def dataplot(i,g,u,title,safeimg):
    plt.figure(i)
    t = [np.log(i) for i in horizons]
    plt.plot(t, g, label = "thompson-sampling")
    plt.plot(t, u, label = "thompson-samplint-with-hint")
    plt.xlabel('Horizons')
    plt.ylabel('Regret')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig(safeimg)

dataplot(1,thompson,thompson_hint,"for instances/i-1.txt","T2instance-1.png")
dataplot(2,thompson1,thompson_hint1,"for instances/i-2.txt","T2instance-2.png")
dataplot(3,thompson2,thompson_hint2,"for instances/i-3.txt","T2instance-3.png")


