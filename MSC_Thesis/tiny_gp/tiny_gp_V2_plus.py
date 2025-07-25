# tiny genetic programming plus, by © moshe sipper, www.moshesipper.com
# graphic output, dynamic progress display, bloat-control option 
# need to install https://pypi.org/project/graphviz/

from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/bryan/MSC_Thesis/Player_Inputs/Scripts/Gathering_Input"))
from rominfo import *
import retro
import time
import matplotlib.pyplot as plt
from IPython.display import Image, display
from graphviz import Digraph, Source 

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
ELITISM         = 2    # maximal number of best runs that pass to the next generation
DEATHPUNISHMENT = 20   # decide how much a death would be punished for
MAX_STEPS       = 300  # maximal steps the agent can do with one action
FINISH          = 4820 # The x position of the worlds finish or just a goal for your mario agent
GENERATIONS     = 250  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate 
PROB_MUTATION   = 0.2  # per-node mutation probability 

def do(env, x, y, view = False, inputs = [] ,states_array = [], distances = []): 
    if (isinstance(x, np.ndarray)):
        i = 0
        iEnd = 0
        Ended = False
        for i in range(y):
            ram = getRam(env)
            marioX, marioY, layer1x, layer1y  = getXY(ram)
            distances.append(marioX)
            if ((marioY > 0) and (marioX < 4820)):
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
                    env.render()  # Render the environment
            else:
                if(Ended == False):
                    iEnd = i
                    Ended = True

        if (Ended == False):
            print(f"{x}, played {i + 1} times")
        else:
            print(f"{x}, played {iEnd} times and ended, would of been {y} times")
    else:
        for action in x:
            # print(action)
            i = 0
            iEnd = 0
            Ended = False
            for i in range(y):
                ram = getRam(env)
                marioX, marioY, layer1x, layer1y  = getXY(ram)
                distances.append(marioX)
                if ((marioY > 0) and (marioX < 4820)):
                    obs, rew, done, _info = env.step(action)  # Play action x, y times in env
                    if(view):
                        saved_inputs = np.array(action)
                        saved_inputs = saved_inputs.astype(int)
                        inputs.append(saved_inputs)

                        ram = getRam(env)
                        state, xi, yi = getInputs(ram)
                        saved_outputs = np.array(state.flatten())
                        saved_outputs = saved_outputs.astype(int)
                        states_array.append(saved_outputs)
                        env.render()  # Render the environment
                else:
                    if(Ended == False):
                        iEnd = i
                        Ended = True

            if (Ended == False):
                print(f"{action}, played {i + 1} times (loop)")
            else:
                print(f"{action}, played {iEnd} times and ended, would of been {y} times (loop)")

    #print("do has been done")
    print(x)
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

def split(x, y):
    arr = []
    print(f"split x = {x}")
    print(f"split y = {y}") 

    if (isinstance(x, np.ndarray)):
        arr.append(x)
    else:
        for action in x:
            arr.append(action)

    if (isinstance(y, np.ndarray)):
        arr.append(y)
    else:
        for action in y:
            arr.append(action)

    return arr

actions_ag.append(combine(actions_ag[0], actions_ag[3]))
actions_ag.append(combine(actions_ag[0], actions_ag[4]))
actions_ag.append(combine(actions_ag[1], actions_ag[3]))
actions_ag.append(combine(actions_ag[1], actions_ag[4]))
actions_ag.append(combine(actions_ag[2], actions_ag[3]))
actions_ag.append(combine(actions_ag[2], actions_ag[4]))
actions_ag.append(combine(actions_ag[5], actions_ag[3]))
actions_ag.append(combine(actions_ag[5], actions_ag[4]))
actions_ag.append(combine(actions_ag[0], combine(actions_ag[1], actions_ag[3])))
actions_ag.append(combine(actions_ag[0], combine(actions_ag[1], actions_ag[4])))
actions_ag.append(combine(actions_ag[1], combine(actions_ag[5], actions_ag[3])))
actions_ag.append(combine(actions_ag[1], combine(actions_ag[5], actions_ag[4])))

FUNCTIONS = [do, split]
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
    def __init__(self, data = None, left = None, right = None, depth = None):
        self.data  = data
        self.left  = left
        self.right = right
        self.depth = depth
        
    def node_label(self): # string label
        if (isinstance(self.data, np.ndarray) != True):
            if (self.data in FUNCTIONS):
                return self.data.__name__
            else: 
                return str(self.data)
        else: 
            return str(self.data)
    
    def draw(self, dot, count): # dot & count are lists in order to pass "by reference" 
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)
        
    def draw_tree(self, fname, footer):
        dot = [Digraph()]
        dot[0].attr(kw='graph', label = footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename = fname + ".gv", format="png").render()
        display(Image(filename = fname + ".gv.png"))

    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, env, Tdistances): 
        # print(f"self.data = {self.data}, combine = {combine}, do = {do}")
        if (isinstance(self.data, np.ndarray) != True): 
            if (self.data == do):
                return self.data(env, self.left.compute_tree(env, Tdistances), self.right.data, distances = Tdistances)
            else: 
                return self.data(self.left.compute_tree(env, Tdistances), self.right.compute_tree(env, Tdistances))
        else: return self.data
            
    def random_tree(self, grow, max_depth, depth = 0, init = False): # create random tree using either grow or full method
        if depth == 0 and init:
            self.data = FUNCTIONS[0]
            self.depth = depth
        elif self.depth == 0 and grow:
            self.data = FUNCTIONS[0]
            self.depth = depth
        elif depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]

        if ((isinstance(self.data, np.ndarray) or type(self.data) == int) == False): # if self.data is a function
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

    def mutation(self, count): # note: count is list, so it's passed "by reference"
        print("mutation in progress")
        count[0] -= 1
        # self.print_tree()
        if count[0] <= 1 and random() < PROB_MUTATION: # mutate at this node
            print("mutation happend")
            self.random_tree(grow = True, max_depth = 2)
        else:
            if self.left:  
                print("mutation when left")
                self.left.mutation(count)
            if self.right and (type(self.right.data) != int): 
                print("mutation when right")
                self.right.mutation(count) 
        
#    def depth(self):     
#        if self.data in TERMINALS: return 0
#        l = self.left.depth()  if self.left  else 0
#        r = self.right.depth() if self.right else 0
#        return 1 + max(l, r)

    def size(self): # tree size in nodes
        if (isinstance(self.data, np.ndarray) or type(self.data) == int): return 1
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
                print("return subtree")
                return self.build_subtree()
            else: # glue subtree here
                print("glue subtree")
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: 
                print("scan_tree when left")
                ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1 and (type(self.right.data) != int): 
                print("scan_tree when right")
                ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            print("crossover happend")
            #print("___________second_before_scan_tree____________")
            #other.print_tree()
            #print(f"other size = {other.size()}")
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            #print("___________second_after_scan_tree____________")
            #second.print_tree()

            # randA = int(self.size() / 2)
            # if(randA < 3):
            #    randA = 3
            randA = 3
            print(f"the size of self is {randA}")
            self.scan_tree([randint(randA, self.size())], second) # 2nd subtree "glued" inside 1st tree

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
            t.random_tree(grow = True, max_depth = md, init = True) # grow
            pop.append(t) 
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md, init = True) # full
            pop.append(t) 
    return pop

def fitness(individual, pop, gen):
    env = retro.make(game="SuperMarioWorld-Snes", state=level, scenario=None, obs_type=retro.Observations.IMAGE)
    obs = env.reset()
    distances = []
    startTime = time.perf_counter()

    individual.compute_tree(env, distances)
    ram = getRam(env)
    marioX, marioY, layer1x, layer1y  = getXY(ram)
    env.close()

    endTime = time.perf_counter()
    elapsedTime = endTime - startTime

    punishments = len(distances) / 100
    if ((marioY == 0)):
        punishments += DEATHPUNISHMENT
    fitness = (100 * (((max(distances) - punishments) - elapsedTime) / FINISH))

    print(f"max(distances) = {max(distances)} for pop {pop} in gen {gen}")
    print(f"len(distances) = {len(distances)}")
    print(f"fitness = {fitness}")
    print(f"end marioX = {marioX}, marioY = {marioY}")
    print(f"elapsedTime = {elapsedTime} seconds")

    if (max(distances) >= FINISH):
        return 100
    return fitness
                
def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]]) 
            
def prepare_plots():
    fig, axarr = plt.subplots(2, sharex=True)
    fig.canvas.set_window_title('EVOLUTIONARY PROGRESS V2')
    fig.subplots_adjust(hspace = 0.5)
    axarr[0].set_title('max fitness', fontsize=14)
    axarr[1].set_title('mean size', fontsize=14)
    plt.xlabel('generation', fontsize=18)
    plt.ion() # interactive mode for plot
    axarr[0].set_xlim(0, GENERATIONS)
    axarr[0].set_ylim(0, 100) # fitness range
    xdata = []
    ydata = [ [], [] ]
    line = [None, None]
    line[0], = axarr[0].plot(xdata, ydata[0], 'b-') # 'b-' = blue line    
    line[1], = axarr[1].plot(xdata, ydata[1], 'r-') # 'r-' = red line
    return axarr, line, xdata, ydata

def plot(axarr, line, xdata, ydata, gen, pop, fitness, max_mean_size):
    xdata.append(gen)
    ydata[0].append(max(fitness))
    line[0].set_xdata(xdata)
    line[0].set_ydata(ydata[0])
    sizes = [ind.size() for ind in pop]
    if mean(sizes) > max_mean_size[0]:
        max_mean_size[0] = mean(sizes)
        axarr[1].set_ylim(0, max_mean_size[0])
    ydata[1].append(mean(sizes))
    line[1].set_xdata(xdata)
    line[1].set_ydata(ydata[1])
    plt.draw()  
    plt.pause(0.01)

def main():      
    # init stuff
    seed() # init internal state of random number generator
    dataset = generate_dataset()
    population= init_population() 
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], i, "random") for i in range(POP_SIZE)]
    max_mean_size = [0] # track maximal mean size for plotting
    axarr, line, xdata, ydata = prepare_plots()

    # go evolution!
    for gen in range(GENERATIONS):        
        print("gen:", gen, ", has started")   
        nextgen_population=[]

        best_fitnesses = fitnesses.copy()
        best_fitnesses.sort(reverse=True)

        for i in range(POP_SIZE - ELITISM):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)

            print(f"___________parent1_before_crossover_of_pop_{i}_gen_{gen}____________")
            parent1.print_tree()
            print(f"___________parent2_before_crossover_of_pop_{i}_gen_{gen}____________")
            parent2.print_tree()
            parent1.crossover(parent2)
            print(f"___________parent1_after_crossover_of_pop_{i}_gen_{gen}____________")
            parent1.print_tree()
            #parent1.mutation([int(parent1.size() / 4)])
            parent1.mutation([0])
            print(f"___________parent1_after_mutation_of_pop_{i}_gen_{gen}____________")
            parent1.print_tree()
            
            nextgen_population.append(parent1)
        
        for i in range(ELITISM):
            best = population[fitnesses.index(best_fitnesses[i])]
            nextgen_population.append(best)

            print(f"best fitnesse{i + 1} = {best_fitnesses[i]}")
        population=nextgen_population

        fitnesses = [fitness(population[i], i, gen) for i in range(POP_SIZE)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])

            states_array = []
            inputs = []

            print("________________________")
            best_of_run.draw_tree("best_of_run_V2",\
                                  "gen: " + str(gen) + ", fitness: " + str(best_of_run_f))
        plot(axarr, line, xdata, ydata, gen, population, fitnesses, max_mean_size)
        best_of_run.print_tree()

        env = retro.make(game="SuperMarioWorld-Snes", state=level, scenario=None, obs_type=retro.Observations.IMAGE)
        obs = env.reset()
        # best_of_run.replay(env, inputs, states_array)
        #RAM State Array
        final_state_array = np.empty((len(states_array),),dtype=object)
        for i in range(len(states_array)):
            final_state_array[i] = states_array[i]

        #Action Array
        final_action_array = np.empty((len(inputs),),dtype=object)
        for i in range(len(inputs)):
            final_action_array[i] = inputs[i]

        dataset = np.array((final_state_array,final_action_array))
        #np.save("/home/bryan/dissertation/best_run_V2",dataset)

        env.render(close=True)
        env.close()
        
        if best_of_run_f == 100: break  
    
    endrun = "_________________________________________________\nEND OF RUN"
    print(endrun)
    s = "\n\nbest_of_run attained at gen " + str(best_of_run_gen) + " and has f=" + str(round(best_of_run_f,3))
    best_of_run.draw_tree("best_of_run_V2",s)

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
    np.save("/home/bryan/dissertation/best_run_V2_won",dataset)

    env.render(close=True)
    env.close()
    
if __name__== "__main__":
  main()
