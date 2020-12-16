import argparse
import numpy as np
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--grid",help="grid file")
    parser.add_argument("--value_policy",help="grid file")

    args=parser.parse_args()
    gridFile=args.grid
    valuefile=args.value_policy

    fh = open(gridFile)
    lines = fh.readlines()
    fh.close()
    #convert txt to 2d array of grid
    grid = np.array([line.split() for line in lines], dtype=np.int64)
    #numStates
    numStates=0
    start=-1
    end=-1
    numActions=4
    discount=1
    transitions=[]
    reward=-1
    mdptype="epsiodic"
    endreward=100000

    #grid to numbering states initially all -1
    l,b = grid.shape
    states=np.array([[-1]*b]*l, dtype=np.int64)
    #tuple(i,j)
    states_to_index=[]
    #valid  transactions from eachstate
    validTransac=[]

    
    #creeating vlidstates,state to index and index to state map
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j]==2:
                start=numStates
                states[i][j]=numStates
                states_to_index.append((i,j))
                numStates=numStates+1   

            elif grid[i][j]==0:
                states[i][j]=numStates
                states_to_index.append((i,j))
                numStates=numStates+1 
            
            elif grid[i][j]==3:
                states[i][j]=numStates
                states_to_index.append((i,j))
                end=numStates
                numStates=numStates+1 
            else:
                pass

            
    fh=open(valuefile)
    lines=fh.readlines()
    phi = np.array([line.split()[1] for line in lines], dtype=np.int64)
    fh.close()
    ans=[]
    cur=start
    # print(cur)
    while(1):
        if(cur == end):
            break
        i,j=states_to_index[cur]
        action=phi[cur]
        ans.append(action)
        if action==0:
            cur=states[i][j+1]
        elif action==1:
            cur=states[i+1][j]
        elif action==2:
            cur=states[i][j-1]
        elif action==3:
            cur=states[i-1][j]

    # print(ans)
    ansstring=""
    for i in ans:
        if(i==0):
            ansstring=ansstring+"E "
        if(i==1):
            ansstring=ansstring+"S "
        if(i==2):
            ansstring=ansstring+"W "
        if(i==3):
            ansstring=ansstring+"N "

    print(ansstring)


    