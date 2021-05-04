import numpy as np
import os
import torch as T
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, chkpt_dir, env_name, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, env_name)
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)  # pi.shape (1, 4)
        v = self.v(x)    # v.shape (1, 1)
        return (pi, v)

    def save_checkpoint(self):
        print(' ... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_dir, env_name, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, chkpt_dir, env_name, fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state) # probabilities.shape (1, 4)
        probabilities = F.softmax(probabilities, dim=1)  # probabilities.shape (1, 4) but normalized to sum up to 1
        action_probs = T.distributions.Categorical(probabilities) # action_probs - Categorical distribution parameterized by probs
        action = action_probs.sample() # action.shape =(1) , sample() method is selecting a action from the set of actions based on probs
        log_prob = action_probs.log_prob(action) # getting the log prob of the choosen action
        self.log_prob = log_prob
        return action.item()  # action.item() = the index of the choosen action

    def save_model(self):
        self.actor_critic.save_checkpoint()

    def load_model(self):
        self.actor_critic.load_checkpoint()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value
        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss+ critic_loss).backward()   # calculating gradients based on loss
        self.actor_critic.optimizer.step()     # updating parameters based on calculated gradients


def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i]= np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
