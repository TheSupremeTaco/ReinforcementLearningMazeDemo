# Reinforcement Learning Project 1
# Maksym Mostovyy Z235236707
# Reinforcement Learning CAP 6629-001

# Main Things for Maze Program using Reinforcement Learning Algorithms

# Maze Board
#   Can be changed to any user inputted row by column dimensions (default will be 3x3)
#   Start state will be random (by default will be the top left corner [0,0])
#   Goal state will be random (by default will be the bottom right corner [3,3])
#   Agent will have 4 actions available at each state encode as (0-3)
#   up-0 left-1 down-2 right-3
#   There will be no diagonal actions possible
#   Illegal action (i.e.) moving into the border or a wall will result in no state change
#   Each state on the maze will be represented as an array [0,0,0,0]
#   encoded each index represents action value (up-0 left-1 down-2 right-3)
#   The maze board will be an object that takes three possible parameters
#       param1: (int array value i.e. [1,3]) int row and int column size (default int array [3,3])
#       param2: (int array value i.e. [0,0]) starting position (default int array [0,0] bottom left corner)
#       param3: (int array value i.e. [2,2]) goal position (default int array [2,2] top right corner)
#   Creates an object from these three parameters represented visually as:
#   [S, -, -]
#   [-, -, -]
#   [-, -, G]
#   Maze object will have three local variables (.this)
#   currentState: the current state on the board
#   nextState: next possible state on the board
#   possibleActions: possible actions at a state
#   Init all Q(s,a) to 0
#   Init Goal State to 100

# Algorithm Selection
#   Q-Learning: Selecting the next state based on the immediate reward
#   Selects a learning method based on user input (default is Q learning)
#       Q-Learning Pseudo Code
#           nextState = maxAction(currentState.possibleActions)
#           maxAction(currentState.possibleActions) = for (action : possibleActions):
#               if (immediateReward + discount*action > currentMax): currentMax = action

#
#
#       Monte Carlo
#           Pseudo Code

import numpy as np
import matplotlib.pyplot as plt

class gridWorld:
    def __init__(self,heightSize=3,widthSize=3,learningMethod="q",learningRate=0.1,discountFactor=0.9,epochs=100):
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.widthSize = widthSize
        self.heightSize = heightSize
        self.startingState = [0,0]
        self.goalState = [heightSize-1,widthSize-1]
        self.currentState = self.startingState
        self.epochs = epochs
        self.stepsToGo = []
#        self.nextState = nextState
#        self.possibleActions = possibleActions
        self.gridMatrix = np.zeros((heightSize,widthSize), dtype=float)
        self.gridMatrix[self.goalState[0],self.goalState[1]] = 1
        if learningMethod == "q":
           gridWorld.qLearning(self);
        if learningMethod == "m":
            gridWorld.monteCarlo(self)
        print(self.stepsToGo)

    def showPerformanceMetrics(self):
        plt.plot(self.stepsToGo)
        plt.xlabel("Steps to Go")
        plt.ylabel("Number of Episodes")
        plt.title("Q-Learning")
        plt.show()

    def makeAction(self, action):
        #(up - 0 left-1 down-2 right-3)
        self.nextState = self.currentState.copy()
        # Going up
        if (action == 0):
            self.nextState[0] -= 1
        # Going left
        if (action == 1):
            self.nextState[1] -= 1
        # Going down
        if (action == 2):
            self.nextState[0] += 1
        # Going right
        if (action == 3):
            self.nextState[1]+= 1
        #print(action)

    def removeIllegals(self):
        for indexR,row in enumerate(self.qTable):
            for indexC,col in enumerate(row):
                if indexR == 0:
                    self.qTable[indexR,indexC,0] = -1
                if indexR == len(self.qTable) - 1:
                    self.qTable[indexR, indexC, 2] = -1
                if indexC == 0:
                    self.qTable[indexR, indexC, 1] = -1
                if indexC == len(self.qTable)-1:
                    self.qTable[indexR, indexC, 3] = -1

    def qLearning(self):
        #creating a Q table
        #Total number of state and the four possible actions at each state
        self.qTable = np.zeros((self.heightSize,self.widthSize,4),dtype=float)
        #Removing illegal moves (i.e. moving out of bounds)
        self.removeIllegals()
        for count,i in enumerate(range(self.epochs)):
            episodeStepToGo = 0
            #Random starting position
            #self.currentState = np.random.randint(0,2,2).tolist()
            self.currentState = self.startingState
            self.optimalStepsToGo = abs(self.goalState[0]-self.startingState[0])+abs(self.goalState[1]-self.startingState[1])
            while self.currentState != self.goalState:
                episodeStepToGo += 1
                #Max value of next possible action at current state
                possibleCurrentActions = self.qTable[self.currentState[0],self.currentState[1]]
                maxAction = np.flatnonzero(possibleCurrentActions == np.max(possibleCurrentActions))
                if len(maxAction) > 1:
                    maxAction = np.random.choice(maxAction,1)[0]
                self.makeAction(maxAction)
                if self.nextState == self.goalState:
                    reward = 100
                else:
                    reward = 0
                # Q Table Update
                newQTableVal = (1-self.learningRate)*self.qTable[self.currentState[0],self.currentState[1],maxAction] + self.learningRate*(reward + self.discountFactor*np.max(self.qTable[self.nextState[0],self.nextState[1]]))
                self.qTable[self.currentState[0],self.currentState[1],maxAction] = newQTableVal
                self.currentState = self.nextState
            #print(self.qTable)
            print(count)

            self.stepsToGo.append([episodeStepToGo])


    def monteCarlo(self):
        print("test")

testGridWorld = gridWorld(20,20)
testGridWorld.showPerformanceMetrics()