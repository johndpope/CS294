#!/bin/sh

python run_expert.py experts/Reacher-v1.pkl Reacher-v1 --num_rollouts 100 --max_timesteps 1000
