"""
Training loop.
"""

import numpy as np
import torch
import statistics


def train(environment, agent, n_episodes=10000, max_t=1000, solve_score=100.0):
    """ Run training loop.

    Params
    ======
        environment: environment object
        agent: agent object
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        solve_score (float): criteria for considering the environment solved
    """


    stats = statistics.Stats()
    stats_format = 'Buffer: {:6}   NoiseW: {:.4}'

    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = environment.reset()
        # loop over steps
        for t in range(max_t):
            # select an action
            if agent.evaluation_only:  # disable noise on evaluation
                action = agent.act(state, add_noise=False)
            else:
                action = agent.act(state)
            # take action in environment

            # map from 3 continuous actions to 1 discrete action
            action_tmp = action.reshape(4, 3)  # reshape 1x12 dim to 4x3 dim (each of the 4 agents has 3 actions)
            idx = np.argmax(np.abs(action_tmp), axis=1) # pick the largest absolute value for each agent
            is_negative = action_tmp[np.arange(4), idx] < 0 # True if the action value is negative else False
            env_action = (idx * 2) + is_negative # map from continuous action to discrete action
            # network outputs 3 continuous values
            # first value maps to 0,1 (forward, backward)
            # second value maps to 2,3 (spin right, spin left)
            # third value maps to 4,5 (slide left, slide right)
            # largest absolute value is the action chosen
            # mapping only happens here for passing action to environment
            # DEBUG action mapping
            #print('is_negative: {}'.format(is_negative))
            #print('idx: {}'.format(idx))
            #print('env_action: {}'.format(env_action))

            next_state, reward, done = environment.step(env_action)
            # update agent with returned information
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            if all(done):
                break

        # every episode
        buffer_len = len(agent.memory)
        per_agent_rewards = []  # calculate per agent rewards
        for i in range(agent.n_agents):
            per_agent_reward = 0
            for step in rewards:
                per_agent_reward += step[i]
            per_agent_rewards.append(per_agent_reward)
        stats.update(t, [np.sum((per_agent_rewards[0], per_agent_rewards[2]))], [np.sum((per_agent_rewards[1], per_agent_rewards[3]))], i_episode)  # track sum rewards across each team
        stats.print_episode(i_episode, t, stats_format, buffer_len, agent.noise_weight,
                            agent.agents[0].critic_loss, agent.agents[1].critic_loss, agent.agents[2].critic_loss, agent.agents[3].critic_loss,
                            agent.agents[0].actor_loss, agent.agents[1].actor_loss, agent.agents[2].actor_loss, agent.agents[3].actor_loss,
                            agent.agents[0].noise_val, agent.agents[1].noise_val, agent.agents[2].noise_val, agent.agents[3].noise_val,
                            per_agent_rewards[0], per_agent_rewards[1], per_agent_rewards[2], per_agent_rewards[3])

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_epoch(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = 'checkpoints/episode.{}.'.format(i_episode)
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor_local.state_dict(), save_name + str(i) + '.actor.pth')
                torch.save(save_agent.critic_local.state_dict(), save_name + str(i) + '.critic.pth')

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_solve(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = 'checkpoints/solved.'
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor_local.state_dict(), save_name + str(i) + '.actor.pth')
                torch.save(save_agent.critic_local.state_dict(), save_name + str(i) + '.critic.pth')
            break
