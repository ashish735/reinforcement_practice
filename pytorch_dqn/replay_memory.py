import numpy as np 

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size   # 50000 , input_shape = (4, 84, 84)
        self.mem_cntr = 0
        # self.state_memory.shape = (50000, 4, 84, 84)
        self.state_memory = np.zeros((self.mem_size, *input_shape), 
                                        dtype=np.float32)
        # self.new_state_memory.shape = (50000, 4, 84, 84)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        # self.action_memory.shape = (50000,)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state         # state.shape = (4, 84, 84)
        self.action_memory[index] = action       # action.shape = (), action = 3
        self.reward_memory[index] = reward       # reward = 0.0 (i.e reward obtained)
        self.new_state_memory[index] = state_    # state_.shape = (4, 84, 84)
        self.terminal_memory[index] = done       # done = 0 (i.e weather reached a terminal state)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)   # when this is called for the first time, self.mem_cntr= 32, self.mem_size= 50000
        batch = np.random.choice(max_mem, batch_size, replace=False)  # a batch of index values
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones