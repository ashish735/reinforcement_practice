import collections
import numpy as np
import cv2
import gym 

import matplotlib.pyplot as plt
import numpy as np 


def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0,t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, 
                    clip_reward=False, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape   # self.shape = (210, 160, 3), we can also use env.observation_space.high.shape also
        self.frame_buffer = np.zeros_like((2, self.shape), dtype=object)  # self.frame_buffer = array([0, 0], dtype=object)
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):   # self.repeat = 4
            obs, reward, done, info = self.env.step(action)  # no overloading of step method here, repeating the same action 4 times and reaching different observation at each time
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs  # self.frame_buffer[0].shape = (210, 160, 3)
            if done:
                break
        
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])  # max_frame.shape = (210, 160, 3)
        return max_frame, t_reward, done, info   # it is returning to observation method of class PreprocessFrame(gym.ObservationWrapper)

    def reset(self):
        obs = self.env.reset()  # no obverloading of reset method here, obs.shape = (210, 160, 3)
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)
        self.frame_buffer = np.zeros_like((2, self.shape))  # here self.shape = (210, 160, 3) is referring to the shape variable of its own class
        self.frame_buffer[0] = obs   # self.frame_buffer[0].shape = (210, 160, 3)
        return obs   # it is returning to observation method of class PreprocessFrame(gym.ObservationWrapper)

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)  #it was given in the documentation to send env as argument
        self.shape = (shape[2], shape[0], shape[1]) # self.shape = (1, 84, 84)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)      
        # self.observation_space = Box(lowest_possible_value= 0.0, highest_possible_value= 1.0, shape= (1,84,84), dtype=float) 
                                                

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)   # new_frame.shape = (210, 160)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA) # here self.shape = (1, 84, 84) , resized_screen.shape = (84, 84)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape) # new_obs.shape = (1, 84, 84)
        new_obs = new_obs / 255.0
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0), env.observation_space.high.repeat(repeat, axis=0),dtype=np.float32)
        # here the repeat() method is numpy built in method, now self.observation_space = Box(0.0, 1.0, (4, 84, 84), float32) we are including 4 frames of the game                                 
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()  # reset() method is overloaded in RepeatActionAndMaxFrame(gym.Wrapper), observation.shape = (1, 84, 84)
        for _ in range(self.stack.maxlen):   # self.stack.maxlen = 4
            self.stack.append(observation)   
        return np.array(self.stack).reshape(self.observation_space.low.shape) # np.array(self.stack).shape = (4, 1, 84, 84) , self.observation_space.low.shape = (4, 84, 84) , after reshaping shape = (4, 84, 84)

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)  # return array shape is (4, 84, 84)

# clip_rewards, no_ops, fire_first is not used during training
# it is only using during testing
def make_env(env_name, shape=(84, 84, 1), repeat=4, clip_rewards=False, 
              no_ops=0, fire_first=False):
    env = gym.make(env_name)   # Making the env with default settings
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)   # Modifying the default env settings, i.e now we can call env.frame_buffer, env.repeat, overloaded the reset method
    env = PreprocessFrame(shape, env)  ## modifying the env settings
    env = StackFrames(env, repeat)    ## modifying the env settings, overloaded the reset method again here, 
    return env