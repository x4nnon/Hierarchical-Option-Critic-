# PPOC: Proximal Policy Optimization with Option Critic


## Overview

PPOC is an advanced reinforcement learning framework that combines the strengths of Proximal Policy Optimization (PPO) and Option-Critic (OC) architectures. This implementation inspired by the paper **"Learnings Options End-to-End for Continuous Action
Tasks"** by Klissarov et al. However, the implementation is from the paper **"Accelerating Task Generalisation Using Multi-Level Hierarchical Options"** by Cannon and Simsek.

## Key Features

- **Proximal Policy Optimization**: Utilizes PPO for stable and efficient policy updates.
- **Option-Critic Architecture**: Integrates OC to enable dynamic option selection and termination.

- **Stability**: Typically OC has option collapse issues. In this implementation PPO is applied to both levels of the hierarchy. It has been tested at 25 options and no option collapse was observed.

## Installation

To get started with PPOC, clone the repository and install the required dependencies:

```bash
git clone https://github.com/x4nnon/PPOC.git
```

cd PPOC

pip install -r requirements.txt
```

## Usage

The main script for running PPOC is `OC_PPO.py`. The script can be executed with various arguments defined in the `Args` class. Below is an example of how to run the script:

```bash
python3 methods/OC_PPO.py --env_id="procgen:procgen-fruitbot-v0" --total_timesteps=20000000 --num_envs=32
```

### Arguments

- **`exp_name`**: The name of the experiment. This is used for logging and tracking purposes.  
  **Default**: `os.path.basename(__file__)[: -len(".py")]`

- **`seed`**: An integer seed for random number generation to ensure reproducibility of results.  
  **Default**: `0`

- **`torch_deterministic`**: A boolean flag to enable deterministic behavior in PyTorch operations.  
  **Default**: `False`

- **`cuda`**: A boolean flag to enable CUDA for GPU acceleration. Set to `True` to use GPU if available.  
  **Default**: `True`

- **`track`**: A boolean flag to enable tracking of experiments using tools like Weights & Biases.  
  **Default**: `True`

- **`wandb_project_name`**: The name of the Weights & Biases project for logging experiment data.  
  **Default**: `"fracos_StarPilot_A_QuickTest"`

- **`wandb_entity`**: The Weights & Biases entity (user or team) under which the project is logged.  
  **Default**: `"tpcannon"`

- **`env_id`**: The identifier for the environment to be used, e.g., "procgen-bigfish".  
  **Default**: `"procgen-bigfish"`

- **`total_timesteps`**: The total number of timesteps to run the training for.  
  **Default**: `100000`

- **`learning_rate`**: The learning rate for the optimizer.  
  **Default**: `5e-4`

- **`num_envs`**: The number of parallel environments to run.  
  **Default**: `8`

- **`num_steps`**: The number of steps to run in each environment per update.  
  **Default**: `256`

- **`anneal_lr`**: A boolean flag to enable learning rate annealing over time.  
  **Default**: `True`

- **`gamma`**: The discount factor for future rewards.  
  **Default**: `0.999`

- **`num_minibatches`**: The number of minibatches to split the data into for each update.  
  **Default**: `4`

- **`update_epochs`**: The number of epochs to update the policy and value networks.  
  **Default**: `2`

- **`report_epoch`**: The number of steps after which to report evaluation metrics.  
  **Default**: `81920`

- **`anneal_ent`**: A boolean flag to enable annealing of the entropy coefficient.  
  **Default**: `True`

- **`ent_coef_action`**: The coefficient for the entropy term in the action policy loss.  
  **Default**: `0.01`

- **`ent_coef_option`**: The coefficient for the entropy term in the option policy loss.  
  **Default**: `0.01`

- **`clip_coef`**: The coefficient for clipping the policy gradient.  
  **Default**: `0.1`

- **`clip_vloss`**: A boolean flag to enable clipping of the value loss.  
  **Default**: `False`

- **`vf_coef`**: The coefficient for the value function loss.  
  **Default**: `0.5`

- **`norm_adv`**: A boolean flag to normalize advantages. Always set to `True`.  
  **Default**: `True`

- **`max_grad_norm`**: The maximum norm for gradient clipping.  
  **Default**: `0.1`

- **`batch_size`**: The size of the batch for updates. Calculated as `num_envs * num_steps`.  
  **Default**: `0` (calculated during runtime)

- **`minibatch_size`**: The size of each minibatch. Calculated as `batch_size // num_minibatches`.  
  **Default**: `0` (calculated during runtime)

- **`num_iterations`**: The number of iterations to run. Calculated as `total_timesteps // batch_size`.  
  **Default**: `0` (calculated during runtime)

- **`max_ep_length`**: The maximum length of an episode.  
  **Default**: `990`

- **`debug`**: A boolean flag to enable debug mode.  
  **Default**: `False`

- **`proc_start`**: The starting level for procedurally generated environments.  
  **Default**: `1`

- **`start_ood_level`**: The starting level for out-of-distribution evaluation.  
  **Default**: `420`

- **`proc_num_levels`**: The number of levels for procedurally generated environments.  
  **Default**: `32`

- **`proc_sequential`**: A boolean flag to enable sequential levels in procedurally generated environments.  
  **Default**: `False`

- **`max_eval_ep_len`**: The maximum length of an evaluation episode.  
  **Default**: `1001`

- **`easy`**: A boolean flag to enable easy mode for environments.  
  **Default**: `1`

- **`eval_repeats`**: The number of times to repeat evaluations.  
  **Default**: `1`

- **`use_monochrome`**: A boolean flag to use monochrome assets in environments.  
  **Default**: `0`

- **`eval_interval`**: The interval at which to perform evaluations.  
  **Default**: `100000`

- **`eval_specific_envs`**: The number of specific environments to evaluate.  
  **Default**: `32`

- **`eval_batch_size`**: The batch size for evaluations.  
  **Default**: `32`

- **`gae_lambda`**: The lambda parameter for Generalized Advantage Estimation.  
  **Default**: `0.95`

- **`warmup`**: A boolean flag to enable warmup mode if this is off, you will need a trained model.  
  **Default**: `1`

- **`num_options`**: The number of options available to the agent.  
  **Default**: `25`

## Citing PPOC

If you use PPOC in your research, please cite the following papers:

1. **PPOC Paper**: [Learnings Options End-to-End for Continuous Action Tasks]
   - Authors: [Klissarov et al.]

2. **PPO Paper**: [Proximal Policy Optimization Algorithms]
   - Authors: [John Schulman et al.]

3. **OC Paper**: [The Option-Critic Architecture]
   - Authors: [Pierre-Luc Bacon et al.]

4. **Accelerating Task Generalisation Using Multi-Level Hierarchical Options**:
   - Authors: [Cannon and Simsek]


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback, please contact [Your Name] at [Your Email].