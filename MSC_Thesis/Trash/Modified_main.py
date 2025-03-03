import sys
import retro
import numpy as np
import os

import sys
import retro

import numpy as np
from numpy.random import uniform, choice, random
import pickle
import os
import time
from rominfo import *
#from utils import *

radius = 6

def dec2bin(dec):
    binN = []
    while dec != 0:
        binN.append(dec % 2)
        dec = dec / 2
    return binN

def printState(state):
    
    state_n = np.reshape(state.split(','), (2*radius + 1, 2*radius + 1))
    print(state_n)  
    _=os.system("clear")
    mm = {'0':'  ', '1':'$$', '-1':'@@'}
    for i,l in enumerate(state_n):
      line = list(map(lambda x: mm[x], l))
      if i == radius + 1:
        line[radius] = 'X'
      #print(line) 

def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)

def rule():
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1)
    env.reset()
    
    total_reward = 0

    while not env.data.is_done():
        ram = getRam(env)
        state, x, y = getInputs(ram)  # Get the state from RAM
        printState(getState(ram, radius)[0])  # Print the state of the game

        action = env.action_space.sample()  # Random action selection
        ob, rew, done, info = env.step(action)
        total_reward += rew
        env.render()
        #print(info)
    
    return total_reward

def main():
    r = rule()

if __name__ == "__main__":
    main()
