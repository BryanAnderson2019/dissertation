# tiny genetic programming by © moshe sipper, www.moshesipper.com
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/bryan/MSC_Thesis/Player_Inputs/Scripts/Gathering_Input"))
from rominfo import *
import retro

level = "YoshiIsland2" 
buttons = ["B", "Y", "SELECT","START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
actionsArray = [["B"], ["Y"], ["DOWN"], ["LEFT"], ["RIGHT"], ["A"], ["X"]]
actions_ag = []

for action in actionsArray:
    arr = np.array([0] * 12)
    for button in action:
        arr[buttons.index(button)] = 1
    print(arr)
    actions_ag.append(arr)

POP_SIZE        = 30   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
MAX_STEPS       = 300  # maximal steps the agent can do with one action
FINISH          = 4820 # The x position of the worlds finish or just a goal for your mario agent
GENERATIONS     = 250  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate 
PROB_MUTATION   = 0.2  # per-node mutation probability 

def do(env, x, y, view = False, inputs = [] ,states_array = []): 
    i = 0
    ram = getRam(env)
    marioX, marioY, layer1x, layer1y  = getXY(ram)
    obs, rew, done, _info = env.step(np.array([0] * 12))  # Play action x, y times in env
    if(view):
        inputs.append(np.array([0] * 12))
    
        ram = getRam(env)
        state, xi, yi = getInputs(ram)
        saved_outputs = np.array(state.flatten())
        saved_outputs = saved_outputs.astype(int)
        states_array.append(saved_outputs)

    if ((marioY > 0) and (rew != 100)):
        for i in range(y):
            ram = getRam(env)
            marioX, marioY, layer1x, layer1y  = getXY(ram)

            if ((marioY > 0) and (rew != 100)):
                obs, rew, done, _info = env.step(x)  # Play action x, y times in env
                if(view):
                    saved_inputs = np.array(x)
                    saved_inputs = saved_inputs.astype(int)
                    inputs.append(saved_inputs)

                    ram = getRam(env)
                    state, xi, yi = getInputs(ram)
                    saved_outputs = np.array(state.flatten())
                    saved_outputs = saved_outputs.astype(int)
                    states_array.append(saved_outputs)
            else:
                print(f"{x}, played {i} times and ended")
                return x

            if(view):
                env.render()  # Render the environment
    print(f"{x}, played {i} times")
    #print("do has been done")
    return x

def combine(x, y): 
    arr = np.array([0] * 12)
    for i in range(12):
        if x[i] + y[i] > 1:
            arr[i] = 1
        else:
            arr[i] = x[i] + y[i]

    # print("combine has been done")
    return arr
FUNCTIONS = [do, combine]
TERMINALS = actions_ag
print(len(TERMINALS))

def target_func(x): # evolution's target
    return x*x*x*x + x*x*x + x*x + x + 1

def generate_dataset(): # generate 101 data points from target_func
    dataset = []
    for x in range(-100,101,2): 
        x /= 100
        dataset.append([x, target_func(x)])
    return dataset

class GPTree:
    def __init__(self, data = None, left = None, right = None):
        self.data  = data
        self.left  = left
        self.right = right
        
    def node_label(self): # string label
        if (isinstance(self.data, np.ndarray) != True):
            if (self.data in FUNCTIONS):
                return self.data.__name__
            else: 
                return str(self.data)
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, env): 
        # print(f"self.data = {self.data}, combine = {combine}, do = {do}")
        if (isinstance(self.data, np.ndarray) != True): 
            if (self.data == do):
                return self.data(env, self.left.compute_tree(env), self.right.data)
            else: 
                return self.data(self.left.compute_tree(env), self.right.compute_tree(env))
        else: return self.data
            
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth == 0:
            self.data = FUNCTIONS[0]
        elif depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if ((isinstance(self.data, np.ndarray) or type(self.data) == int) == False):
            if (self.data == do):
                self.left = GPTree()          
                self.left.random_tree(grow, max_depth, depth = depth + 1)
                self.right = GPTree() 
                self.right.data = randint(1, MAX_STEPS)
            else:
                self.left = GPTree()          
                self.left.random_tree(grow, max_depth, depth = depth + 1)            
                self.right = GPTree()
                self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION: # mutate at this node
            print("mutation happend")
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation() 

    def size(self): # tree size in nodes
        if (isinstance(self.data, np.ndarray) or type(self.data) == int): return 0
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self): # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
    
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"
        print(f"count = {count[0]}")
        print(f"self.data = {self.data}")
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1 and (isinstance(self.data, np.ndarray) or type(self.data) == int): ret = self.left.scan_tree(count, second)  
            if self.right and count[0] > 1 and (isinstance(self.data, np.ndarray) or type(self.data) == int): ret = self.right.scan_tree(count, second)  
            return ret
    
    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            print("crossover happend")
            print("___________second_before_scan_tree____________")
            other.print_tree()
            print(f"other size = {other.size()}")
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            print("___________second_after_scan_tree____________")
            # second.print_tree()
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree

    def replay(self, env, Tinputs, Tstates_array): 
        # print(f"self.data = {self.data}, combine = {combine}, do = {do}")
        if (isinstance(self.data, np.ndarray) != True): 
            if (self.data == do):
                return self.data(env, self.left.replay(env, Tinputs, Tstates_array), self.right.data, view = True, inputs = Tinputs, states_array = Tstates_array)
            else: 
                return self.data(self.left.replay(env, Tinputs, Tstates_array), self.right.replay(env, Tinputs, Tstates_array))
        else: return self.data
# end class GPTree
                   
def init_population(): # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t) 
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t) 
    return pop

def fitness(individual, pop, gen):
    env = retro.make(game="SuperMarioWorld-Snes", state=level, scenario=None, obs_type=retro.Observations.IMAGE)
    obs = env.reset()
    ram = getRam(env)
    marioX, marioY, layer1x, layer1y  = getXY(ram)
    obs, rew, done, _info = env.step(np.array([0] * 12))
    distances = [marioX]

    # while ((marioY > 0) and (rew != 100)):
    individual.compute_tree(env)

    ram = getRam(env)
    marioX, marioY, layer1x, layer1y  = getXY(ram)
    obs, rew, done, _info = env.step(np.array([0] * 12))
    distances.append(marioX)
    # print(f"marioX = {marioX}, marioY = {marioY}")

    print(f"max(distances) = {max(distances)} for pop {pop} in gen {gen}")
    print(f"len(distances) = {len(distances)}")
    print(f"end marioX = {marioX}, marioY = {marioY}")
    env.close()
    return 1 / (1 + abs(FINISH - max(distances)))
                
def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]]) 
            
def main():      
    # init stuff
    seed() # init internal state of random number generator
    dataset = generate_dataset()
    population= init_population() 
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], i, "random") for i in range(POP_SIZE)]

    # go evolution!
    for gen in range(GENERATIONS):
        print("gen:", gen, ", has started")        
        nextgen_population=[]
        for i in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            print(f"___________parent1_before_crossover_of_pop_{i}_gen_{gen}____________")
            parent1.print_tree()
            print(f"___________parent2_before_crossover_of_pop_{i}_gen_{gen}____________")
            parent2.print_tree()
            parent1.crossover(parent2)
            print(f"___________parent1_after_crossover_of_pop_{i}_gen_{gen}____________")
            parent1.print_tree()
            parent1.mutation()
            print(f"___________parent1_after_mutation_of_pop_{i}_gen_{gen}____________")
            parent1.print_tree()
            
            nextgen_population.append(parent1)
        population=nextgen_population
        fitnesses = [fitness(population[i], i, gen) for i in range(POP_SIZE)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])

            states_array = []
            inputs = []

            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses),3), ", best_of_run:") 
            best_of_run.print_tree()

            env = retro.make(game="SuperMarioWorld-Snes", state=level, scenario=None, obs_type=retro.Observations.IMAGE)
            obs = env.reset()
            best_of_run.replay(env, inputs, states_array)
            #RAM State Array
            final_state_array = np.empty((len(states_array),),dtype=object)
            for i in range(len(states_array)):
                final_state_array[i] = states_array[i]

            #Action Array
            final_action_array = np.empty((len(inputs),),dtype=object)
            for i in range(len(inputs)):
                final_action_array[i] = inputs[i]

            dataset = np.array((final_state_array,final_action_array))
            np.save("/home/bryan/dissertation/best_run",dataset)

            env.render(close=True)
            env.close()

        if best_of_run_f == 1: break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f,3)))
    best_of_run.print_tree()

    env = retro.make(game="SuperMarioWorld-Snes", state=level, scenario=None, obs_type=retro.Observations.IMAGE)
    obs = env.reset()
    best_of_run.replay(env, inputs, states_array)
    #RAM State Array
    final_state_array = np.empty((len(states_array),),dtype=object)
    for i in range(len(states_array)):
        final_state_array[i] = states_array[i]

    #Action Array
    final_action_array = np.empty((len(inputs),),dtype=object)
    for i in range(len(inputs)):
        final_action_array[i] = inputs[i]

    dataset = np.array((final_state_array,final_action_array))
    np.save("/home/bryan/dissertation/best_run_won",dataset)

    env.render(close=True)
    env.close()
    
    
if __name__== "__main__":
  main()
