# Kuka- Reinforcement Learning in pybullet
###### This repo contains the code for the implementation of PPO algorithm using GAE value estimation, on Kuka robot in pybullet simulation.
##### Implementation details
###### The states of the robot is a h,w,c image feed which is feed into the actor crtic network to outpout the action and value of the corresponding state. 
###### The policy of the agent is updated using the proximal policy optimization algorithm (https://arxiv.org/abs/1707.06347). The value funtion is updated using Generalized advantage estimation method (https://arxiv.org/pdf/1506.02438.pdf).

