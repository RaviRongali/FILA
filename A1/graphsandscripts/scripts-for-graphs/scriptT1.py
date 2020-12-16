import os
import subprocess

instances=["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
algo=["epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
epsilon=0.02
horizons=[100, 400, 1600, 6400, 25600, 102400]
seeds=[i for i in range(50)]


f = open("outputDataT1.txt", "w")
for inst in instances:
    for al in algo:
        for hr in range(len(horizons)):
            total=0.0
            for sd in seeds:
                # if hr<2 and sd<3:
                cmd="python3 bandit.py --instance "+ inst+ " --algorithm " +al+ " --randomSeed " +str(sd)+ " --epsilon " +str(epsilon)+ " --horizon " +str(horizons[hr])
                k = subprocess.check_output(cmd, shell=True).decode("utf-8").rstrip()
                print(k)
                f.write(k)
                f.write("\n")
f.close()                    


