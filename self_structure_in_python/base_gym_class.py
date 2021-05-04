class Env(object):
    def step(self, action):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class GymAtari(Env):
    def __init__(self, start):
        self.initial_state = start
        self.state = start

    def step(self, action):    # self refers to the object of GymAtari
        self.state +=action
        return self.state

    def reset(self):
        self.state = self.initial_state
        return self.state

class Wrapper(Env):    # Wrapper is a class which takes objects of another class,
    def __init__(self, env):  # here env is object of class GymAtari
        self.env = env

    def step(self, action):
        return self.env.step(action)    # self refers to the object of Wrapper

    def reset(self):
        return self.env.reset()

class ObWrapper(Wrapper):
    def step(self, action):
        a = self.env.step(action)    # self refers to the object of PreProcess or Stack
        return self.observation(a)

    def observation(self, observation):
        raise NotImplementedError

