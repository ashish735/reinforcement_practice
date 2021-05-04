import base_gym_class

class RepeatAction(base_gym_class.Wrapper):
    def __init__(self, env, repeat):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        for i in range(self.repeat):      # self refers to the object of RepeatAction
            a = self.env.step(action)
        return a

class PreProcess(base_gym_class.ObWrapper):
    def __init__(self, env):
        super(PreProcess, self).__init__(env)  # self refers to the object of PreProcess

    def observation(self, obs):
        obs = obs * 20
        return obs

class Stack(base_gym_class.ObWrapper):
    def __init__(self, env):
        super(Stack, self).__init__(env)

    def observation(self, obs):
        obs = obs * 20
        return obs