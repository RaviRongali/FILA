import argparse
import numpy as np



def transitionf(s1,ac,s2,r,p):
        return str("transition" + " " + str(s1) + " " + str(ac) + " " + str(s2) + " " + str(r) + " " + str(p))



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--grid",help="grid file")
    args=parser.parse_args()
    gridFile=args.grid

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
    discount=0.99
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

    ### making transactions for valid states
    for eachstate in range(numStates):
        i,j=states_to_index[eachstate]
        if states[i][j]!=end:
            #for each i,j tuple(ac,s2,reward)
            # Actions 0 -> E, 2-> S, 3-> W, 3-> N
            valid=[]
            #given maze all are having boundaries as 1 that are not valid
            ##north
            if(i>0):
                #if N state is end state
                Nstate=states[i-1][j]
                if Nstate!=-1:
                    if Nstate==end:
                        valid.append((3,Nstate,endreward))
                    else:
                        valid.append((3,Nstate,reward))
            
            if(i<l-1):
                #if Sstate is end state
                Sstate=states[i+1][j]
                if Sstate!=-1:
                    if Sstate==end:
                        valid.append((1,Sstate,endreward))
                    else:
                        valid.append((1,Sstate,reward))

            if(j>0):
                #if Wstate is end state
                Wstate=states[i][j-1]
                if Wstate!=-1:
                    if Wstate==end:
                        valid.append((2,Wstate,endreward))
                    else:
                        valid.append((2,Wstate,reward))

            if(j<b-1):
                #if Estate is end state
                Estate=states[i][j+1]
                if Estate!=-1:
                    if Estate==end:
                        valid.append((0,Estate,endreward))
                    else:
                        valid.append((0,Estate,reward))

            validTransac.append(valid)
            for move in valid:
                transitions.append(transitionf(states[i,j],move[0],move[1],move[2],1.0))
                
      
    print("numStates {}".format(numStates))
    print("numActions {}".format(4))
    print("start {}".format(start))
    print("end {}".format(end))
    for i in transitions:
        print(i)
    print("mdptype {}".format(mdptype))
    print("discount {}".format(discount))


    
