import gym
import numpy as np
from ddpg_torch import Agent, plot_learning_curve

if __name__=='__main__':
    env= gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001,
                input_dims=env.observation_space.shape, tau=0.001, # input_dims.shape (8,)
                batch_size=64, fc1_dims=400, fc2_dims=300,
                n_actions=env.action_space.shape[0])         # n_actions.shape (2,) real valued b/w -1 to 1
    load_checkpoint = True

    if load_checkpoint:
        agent.load_models()

    n_games = 5
    filename= 'LunarLander'
    figure_file = '.\\plots\\'+filename+'.png'

    best_score = env.reward_range[0]  # least possible reward in the game
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if not load_checkpoint:
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
            else:
                env.render()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score, 'best score %.1f' % best_score)
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)