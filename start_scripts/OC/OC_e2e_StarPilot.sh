#!/bin/bash

# CREATE TRAJECTORIES

ENV_ID="procgen:procgen-starpilot-v0"
# "procgen:procgen-coinrun-v0"
WANDB_PROJECT_NAME="e2e_OC_StarPilot_warmup"


python3 methods/OC.py --warmup=1 --env_id="$ENV_ID" --total_timesteps=20000000 \
 --wandb_project_name="$WANDB_PROJECT_NAME" --proc_start=1 --proc_num_levels=1


WANDB_PROJECT_NAME="PPO_StarPilot_comparison_A"

X_VALUES=(1 2 3)  # Specify the values for the seed here
# Loop over each combination of x and y


for x in "${X_VALUES[@]}"; do

    python3 methods/OC.py --warmup=0 --env_id="$ENV_ID" --total_timesteps=5000000\
     --wandb_project_name="$WANDB_PROJECT_NAME" --proc_start=1 --proc_num_levels=1 --seed="$x"

done