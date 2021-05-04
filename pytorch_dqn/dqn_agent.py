import numpy as np 
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                mem_size, batch_size, chkpt_dir, eps_min=0.01, eps_dec=5e-7, replace=1000,
                algo=None, env_name=None):
        self.gamma = gamma                  # 0.99
        self.epsilon = epsilon              # 1.0
        self.lr = lr                        # 0.0001
        self.n_actions = n_actions          # 6
        self.input_dims = input_dims        # (4, 84, 84)
        self.batch_size = batch_size        # 32
        self.eps_min = eps_min              # 0.1
        self.eps_dec = eps_dec              # 1e-05
        self.replace_target_cnt = replace   # 1000
        self.algo = algo                    # 'DQNAgent'
        self.env_name = env_name            #  'PongNoFrameskip-v4'
        self.chkpt_dir = chkpt_dir          #  .\\models\\
        self.action_space = [i for i in range(self.n_actions)] # [0, 1, 2, 3, 4, 5]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, 
                                    input_dims=self.input_dims, 
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions, 
                                    input_dims=self.input_dims, 
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action 

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())    # load_state_dict and state_dict are inbuilt functions of torch

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]   # self.q_eval.forward(states).shape = (32, 6), q_pred.shape = 32
        q_next = self.q_next.forward(states_).max(dim=1)[0]      # self.q_next.forward(states_).shape = (32, 6), q_next.shape = 32

        temp_dones = dones.bool()
        q_next[temp_dones] = 0.0   # as reward for terminal state is 0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()                        
        