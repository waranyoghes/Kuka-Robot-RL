# -*- coding: utf-8 -*-
"""
Created on Sun Jan 8 12:06:46 2022

@author: waran
"""

import matplotlib.pyplot as plt
import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import numpy as np
import random
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p
from model import ActorCritic
from utils import get_screen, collect_trajectories, calc_returns,concat_all, eval_pic


env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20)
env.cid = p.connect(p.DIRECT)
action_space = spaces.Box(low=-1, high=1, shape=(5,1))

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_space.shape[0]





def main():
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    policy=ActorCritic(state_size=(screen_height, screen_width),
              action_size=action_size,
              shared_layers=[128, 64],
              critic_hidden_layers=[64],
              actor_hidden_layers=[64],
              init_type='xavier-uniform',
              seed=0).to(device)


    optimizer = optim.Adam(policy.parameters(), lr=2e-4)
    PATH = 'policy_ppo.pt'
    
    best_mean_reward = None
    scores_window = deque(maxlen=100)  # last 100 scores
    discount = 0.993
    epsilon = 0.07
    beta = .01
    opt_epoch = 10
    season = 1000000
    batch_size = 128
    tmax = 1000 #env episode steps
    save_scores = []
    start_time = timeit.default_timer()
    for s in range(season):
        policy.eval()
        old_probs_lst, states_lst, actions_lst, rewards_lst, values_lst, dones_list = collect_trajectories(envs=env,
                                                                                                       policy=policy,
                                                                                                       tmax=tmax,
                                                                                                       nrand = 5)

        season_score = rewards_lst.sum(dim=0).item()
        scores_window.append(season_score)
        save_scores.append(season_score)
    
        gea, target_value = calc_returns(rewards = rewards_lst,
                                     values = values_lst,
                                     dones=dones_list)
        gea = (gea - gea.mean()) / (gea.std() + 1e-8)
        policy.train()
        old_probs_lst = concat_all(old_probs_lst)
        states_lst = concat_all(states_lst)
        actions_lst = concat_all(actions_lst)
        rewards_lst = concat_all(rewards_lst)
        values_lst = concat_all(values_lst)
        gea = concat_all(gea)
        target_value = concat_all(target_value)
        
        # gradient ascent step
        n_sample = len(old_probs_lst)//batch_size
        idx = np.arange(len(old_probs_lst))
        np.random.shuffle(idx)
        for epoch in range(opt_epoch):
          for b in range(n_sample):
            ind = idx[b*batch_size:(b+1)*batch_size]
            g = gea[ind]
            tv = target_value[ind]
            actions = actions_lst[ind]
            old_probs = old_probs_lst[ind]

            action_est, values = policy(states_lst[ind])
            sigma = nn.Parameter(torch.zeros(action_size))
            dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
            log_probs = dist.log_prob(actions)
            log_probs = torch.sum(log_probs, dim=-1)
            entropy = torch.sum(dist.entropy(), dim=-1)

            ratio = torch.exp(log_probs - old_probs)
            ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            L_CLIP = torch.mean(torch.min(ratio*g, ratio_clipped*g))
            # entropy bonus
            S = entropy.mean()
            # squared-error value function loss
            L_VF = 0.5 * (tv - values).pow(2).mean()
            # clipped surrogate
            L = -(L_CLIP - L_VF + beta*S)
            optimizer.zero_grad()
            # retain_graph=True on the backward pass
            L.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            del(L)

    
        epsilon*=.999
    
    
        beta*=.998

        mean_reward = np.mean(scores_window)
        
    
        if best_mean_reward is None or best_mean_reward < mean_reward:
                # For saving the model
                torch.save({
                        'policy_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epsilon': epsilon,
                        'beta': beta
                        }, PATH)
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
        if s>=25 and mean_reward>50:
            print('Environment solved in {:d} seasons!\tAverage Score: {:.2f}'.format(s+1, mean_reward))
            break
    print('Average Score: {:.2f}'.format(mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    
    env.close()
    eval_pic(envs=env, policy=policy)
        






if __name__ == "__main__":
    main()
