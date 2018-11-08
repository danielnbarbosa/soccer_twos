"""
Statistics to track agent performance.
"""


import time
from collections import deque
import numpy as np
from tensorboardX import SummaryWriter


class Stats():
    def __init__(self):
        self.red_score = None
        self.red_avg_score = None
        self.red_std_dev = None
        self.red_scores = []                         # list containing scores from each episode
        self.red_avg_scores = []                     # list containing average scores after each episode
        self.red_scores_window = deque(maxlen=100)   # last 100 scores
        self.red_best_avg_score = -np.Inf            # best score for a single episode

        self.blue_score = None
        self.blue_avg_score = None
        self.blue_std_dev = None
        self.blue_scores = []                         # list containing scores from each episode
        self.blue_avg_scores = []                     # list containing average scores after each episode
        self.blue_scores_window = deque(maxlen=100)   # last 100 scores
        self.blue_best_avg_score = -np.Inf            # best score for a single episode

        self.time_start = time.time()            # track cumulative wall time
        self.total_steps = 0                     # track cumulative steps taken
        self.writer = SummaryWriter()

    def update(self, steps, red_rewards, blue_rewards, i_episode):
        """Update stats after each episode."""
        self.total_steps += steps
        # update red team rewards
        self.red_score = sum(red_rewards)
        self.red_scores_window.append(self.red_score)
        self.red_scores.append(self.red_score)
        self.red_avg_score = np.mean(self.red_scores_window)
        self.red_avg_scores.append(self.red_avg_score)
        self.red_std_dev = np.std(self.red_scores_window)
        # update best average score
        if self.red_avg_score > self.red_best_avg_score and i_episode > 100:
            self.red_best_avg_score = self.red_avg_score
        # update blue team rewards
        self.blue_score = sum(blue_rewards)
        self.blue_scores_window.append(self.blue_score)
        self.blue_scores.append(self.blue_score)
        self.blue_avg_score = np.mean(self.blue_scores_window)
        self.blue_avg_scores.append(self.blue_avg_score)
        self.blue_std_dev = np.std(self.blue_scores_window)
        # update best average score
        if self.blue_avg_score > self.blue_best_avg_score and i_episode > 100:
            self.blue_best_avg_score = self.blue_avg_score

    def is_solved(self, i_episode, solve_score):
        """Define solve criteria."""
        return self.red_avg_score >= solve_score and i_episode >= 100

    def print_episode(self, i_episode, steps, stats_format, buffer_len, noise_weight,
                      critic_loss_0, critic_loss_1, critic_loss_2, critic_loss_3,
                      actor_loss_0, actor_loss_1, actor_loss_2, actor_loss_3,
                      noise_val_0, noise_val_1, noise_val_2, noise_val_3,
                      rewards_0, rewards_1, rewards_2, rewards_3):
        # red team stats are logged to console.  blue would be jsut be the opposite sign.
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Reward: {:8.3f}  |  '.format(i_episode, self.red_avg_score, self.red_best_avg_score, self.red_std_dev, steps, self.red_score)
        print('\r' + common_stats + stats_format.format(buffer_len, noise_weight), end="")
        # log red/blue team and all agent stats to tensorboard
        self.writer.add_scalar('global/buffer_len', buffer_len, i_episode)
        self.writer.add_scalar('global/noise_weight', noise_weight, i_episode)

        self.writer.add_scalar('team_red/reward', self.red_score, i_episode)
        self.writer.add_scalar('team_red/std_dev', self.red_std_dev, i_episode)
        self.writer.add_scalar('team_red/avg_reward', self.red_avg_score, i_episode)

        self.writer.add_scalar('team_blue/reward', self.blue_score, i_episode)
        self.writer.add_scalar('team_blue/std_dev', self.blue_std_dev, i_episode)
        self.writer.add_scalar('team_blue/avg_reward', self.blue_avg_score, i_episode)

        self.writer.add_scalar('goalie_red/critic_loss', critic_loss_0, i_episode)
        self.writer.add_scalar('goalie_red/actor_loss', actor_loss_0, i_episode)
        self.writer.add_scalar('goalie_red/noise_val', noise_val_0[0], i_episode)
        self.writer.add_scalar('goalie_red/reward', rewards_0, i_episode)

        self.writer.add_scalar('goalie_blue/critic_loss', critic_loss_1, i_episode)
        self.writer.add_scalar('goalie_blue/actor_loss', actor_loss_1, i_episode)
        self.writer.add_scalar('goalie_blue/noise_val', noise_val_1[0], i_episode)
        self.writer.add_scalar('goalie_blue/reward', rewards_1, i_episode)

        self.writer.add_scalar('striker_red/critic_loss', critic_loss_2, i_episode)
        self.writer.add_scalar('striker_red/actor_loss', actor_loss_2, i_episode)
        self.writer.add_scalar('striker_red/noise_val', noise_val_2[0], i_episode)
        self.writer.add_scalar('striker_red/reward', rewards_2, i_episode)

        self.writer.add_scalar('striker_blue/critic_loss', critic_loss_3, i_episode)
        self.writer.add_scalar('striker_blue/actor_loss', actor_loss_3, i_episode)
        self.writer.add_scalar('striker_blue/noise_val', noise_val_3[0], i_episode)
        self.writer.add_scalar('striker_blue/reward', rewards_3, i_episode)
        # DEBUG rewards for each agent
        print('')
        #n_secs = int(time.time() - self.time_start)
        #print('RG: {:.3f}   BG: {:.3f}   RS: {:.3f}   BS: {:.3f}   | Secs: {:6}'.format(rewards_0, rewards_1, rewards_2, rewards_3, n_secs))

    def print_epoch(self, i_episode, stats_format, *args):
        n_secs = int(time.time() - self.time_start)
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Secs: {:6}      |  '.format(i_episode, self.red_avg_score, self.red_best_avg_score, self.red_std_dev, self.total_steps, n_secs)
        print('\r' + common_stats + stats_format.format(*args))

    def print_solve(self, i_episode, stats_format, *args):
        self.print_epoch(i_episode, stats_format, *args)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))
