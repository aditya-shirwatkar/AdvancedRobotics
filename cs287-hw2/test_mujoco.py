import mujoco_py
import gym
env = gym.make('Hopper-v2')  # or 'Humanoid-v2' 

s = env.reset()
done = False
i = 0
while not done:
	env.render()
	i+=1
env.close()
