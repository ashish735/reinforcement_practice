import gym
import numpy as np
from actor_critic import Agent, plot_learning_curve


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    best_score = -np.inf
    load_checkpoint = True

    agent = Agent(gamma= 0.99, lr= 5e-6, input_dims=[8], n_actions=4, fc1_dims=2048, fc2_dims=1536
                    ,chkpt_dir= '.\\models\\', env_name='Lunar_Practice')
    
    if load_checkpoint:
        agent.load_model()

    n_games = 5
    fname = 'ACTOR_CRITIC_'+'lunar_practice'
    figure_file = '.\\plots\\'+fname+'.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            else:
                env.render()
            observation= observation_
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score,
                'best score %.1f' % best_score)
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_model()
            best_score = avg_score
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(scores, x, figure_file)

