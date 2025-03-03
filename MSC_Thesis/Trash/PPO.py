import retro
import gym

env = retro.make(game = "SuperMarioWorld-Snes",state='YoshiIsland2',obs_type = retro.Observations.IMAGE)
obs = env.reset()

print(obs.shape)
done = False

while not done:
    obs, reward, done, info = env.step(env.action_space.sample())