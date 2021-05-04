import numpy as np
from dqn_agent import DQNAgent
from util import make_env, plot_learning_curve
from gym import wrappers

if __name__ == "__main__":
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = True
    n_games = 5
    
    agent = DQNAgent(gamma=0.99, epsilon=0.1, lr=0.0001,
                    input_dims=(env.observation_space.shape),  # we have modified the default env settings to get env.observation_space.shape
                    n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                    batch_size=32, replace=1000, eps_dec=1e-5, 
                    chkpt_dir='.\\models\\', algo='DQNAgent',
                    env_name='PongNoFrameskip-v4')
    if load_checkpoint:
        agent.load_models()

    #env = wrappers.Monitor(env, '.\\video\\', video_callable=lambda episode_id: True, force=True)
    fname= agent.algo+'_'+agent.env_name+'_lr'+str(agent.lr)+'_'+str(n_games)+'games'
    figure_file = '.\\plots\\'+fname+'.png'

    n_steps= 0
    scores, eps_history, steps_array = [], [], []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()  # reset() method is overloaded in class StackFrames(gym.ObservationWrapper) , observation.shape = (4, 84, 84)
                                   # here (1, 84, 84) == (2, 84, 84) == (3, 84, 84) == (4, 84, 84)

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)  # step() method is overloaded in class RepeatActionAndMaxFrame(gym.Wrapper)
                                                                 # here observation_, reward, done, info is reached after repeating the action 4 times
                                                                 # and (2, 84, 84) == (3, 84, 84) == (4, 84, 84) != (1, 84, 84) for the first loop
                                                                 # i.e queue follows a FIFO method, observation_ is stored in (1, 84, 84)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))  # acts like experience replay
                agent.learn()
            else:
                env.render()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print('episode ',i, ' score: ', score,
                'average score %.1f best score %.1f epsilon %.2f' % 
                (avg_score, best_score, agent.epsilon), 
                'steps ', n_steps)
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score
        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)