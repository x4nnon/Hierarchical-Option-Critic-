#!/bin/bash

# CREATE TRAJECTORIES

ENV_ID="procgen-starpilot" # this is running on my local PC which is different from hex naming convention.
# "procgen:procgen-coinrun-v0"
WANDB_PROJECT_NAME="e2e_OC_StarPilot_warmup"

#make sure NN_predict is false in clustering.py


python3 methods/OC_PPO.py --warmup=1 --env_id="$ENV_ID" --total_timesteps=20000000 \
 --wandb_project_name="$WANDB_PROJECT_NAME" --proc_start=1 --proc_num_levels=100 --num_envs=32


WANDB_PROJECT_NAME="PPO_StarPilot_comparison_A"

X_VALUES=(1 2 3)  # Specify the values for the seed here
# Loop over each combination of x and y
for x in "${X_VALUES[@]}"; do
    python3 methods/OC_PPO.py --warmup=0 --env_id="$ENV_ID" --total_timesteps=5000000 --seed="$x"\
     --wandb_project_name="$WANDB_PROJECT_NAME" --proc_start=1 --proc_num_levels=32 --num_envs=32 --eval_batch_size=32 --anneal_lr

done

