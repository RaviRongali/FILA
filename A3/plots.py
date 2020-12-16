import os
print("Task-1 Sarsa(0) with only 4 actions")
print("Arrays of average 10 seeds of timesteps for 170 episodes")
cmd1="python3 gridworld.py --grid gridfile.txt --wind wind.txt --actions 4 --task 1"
os.system(cmd1)

print("Task-2 Sarsa(0) with only 8 actions")
print("Arrays of average 10 seeds of timesteps for 170 episodes")
cmd2="python3 gridworld.py --grid gridfile.txt --wind wind.txt --actions 8 --task 2"
os.system(cmd2)

print("Task-3 Sarsa(0) with only 8 actions and stochastic")
print("Arrays of average 10 seeds of timesteps for 170 episodes")
cmd3="python3 gridworld.py --grid gridfile.txt --wind wind.txt --actions 8 --task 3 --withstocasticity 1"
os.system(cmd3)

print("Task-4 Compare Sarsa(0),ExpectedSarsa,Q-Learning ")
print("Arrays of average 10 seeds of timesteps for 170 episodes")
cmd4="python3 gridworld.py --grid gridfile.txt --wind wind.txt --actions 4 --task 4"
os.system(cmd4)