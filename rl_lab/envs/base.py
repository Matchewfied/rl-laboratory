import gymnasium as gym

"""
Defines the API that every future custom environment should abide by.
This is done for integrability purposes. We are making an instance
of gym.Env, an RL environment provided by OpenAI, because we can debug
our methods on the environments they provide.
"""
class BaseCustomEnv(gym.Env):
    def __init__(self):

        # This is a wrapper around gym.Env. Initialize the gym.Env elements.
        super().__init__()

    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    

