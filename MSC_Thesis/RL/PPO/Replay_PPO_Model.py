import retro
from baselines.common.retro_wrappers import *
from stable_baselines3 import DQN,PPO,A2C
from stable_baselines3.common.policies import obs_as_tensor
import time
import numpy as np
from baselines.common.atari_wrappers import *

env = retro.make(game = "SuperMarioWorld-Snes",use_restricted_actions=retro.Actions.DISCRETE,state = 'YoshiIsland2',obs_type = retro.Observations.IMAGE)

#Frame skip (hold an action for this many frames) and sticky actions
env = StochasticFrameSkip(env,4,0.25)
#scale and turn RGB image to grayscale
env = WarpFrame(env,width=84,height=84,grayscale=True)
blank_action = 0

model = PPO.load("PPO_IMAGE_L1.zip")

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    
    obs, reward, done, info = env.step(action)
    obs, rew, done, info = env.step(blank_action)#Stop it from holding an action, like jump
    
    print(env.get_action_meaning(action))
    env.render()
    if done:
      obs=env.reset()
