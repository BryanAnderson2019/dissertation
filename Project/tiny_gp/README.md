# Tiny Genetic Programming in Python

A minimalistic program implementing Koza-style (tree-based) genetic programming to reach as far as possible in Super Mario World levels. 

**...gp.py** is a basic (and fully functional) version, which produces textual output of the evolutionary progression and evolved trees.

**...plus.py** displays dynamic graphs of error and mean tree size (size = number of nodes), has a bloat-control option, and produces nicer, graphic output (you'll need to install https://pypi.org/project/graphviz/).

| Symbolic | Regression using GP  |
|-------------:|:-------------| 
| Objective | to reach as far as possible in Super Mario World levels |
| Function set | do(main), split(V2), combine(original design), subtract(original design)|   
| Terminal set | playable actions |   
| Fitness | The highest percent that the run reached in a level with punishments to lower the percentage
| Main Paremeters | POP_SIZE (population size), MIN_DEPTH (minimal initial random tree depth), MAX_DEPTH (maximal initial random tree depth), GENERATIONS (maximal number of generations), TOURNAMENT_SIZE (size of tournament for tournament selection), XO_RATE (crossover rate), PROB_MUTATION (per-node mutation probability) |
| Termination | Maximal number of generations reached or an individual with fitness = 100 found (finished a lavel) |

| V2 GP | V2 with our player data |
|-------------|-------------| 
| ![GPTree](https://github.com/BryanAnderson2019/dissertation/blob/main/Project/tiny_gp/Figures/best_of_run_V2.gv_max.png) | ![GPTree2](https://github.com/BryanAnderson2019/dissertation/blob/main/Project/tiny_gp/Figures/best_of_run_V2_PD.gv_max.png) |
|![GP run](https://github.com/BryanAnderson2019/dissertation/blob/main/Project/tiny_gp/Figures/EVOLUTIONARY_PROGRESS_V2_max.png) | ![GP run](https://github.com/BryanAnderson2019/dissertation/blob/main/Project/tiny_gp/Figures/EVOLUTIONARY_PROGRESS_V2_PD_max.png) |
