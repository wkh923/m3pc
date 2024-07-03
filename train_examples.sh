# ***************************
# Pretrain 
# ***************************
# D4RL Hoppper Medium
python research/omtm/train.py +exp_mtm=d4rl_cont wandb.project="omtm_hopper_m" args.seed=0 dataset.env_name=hopper-medium-v2 args.mask_patterns=[AUTO_MASK]


# ***************************
# Finetune
# ***************************
# D4RL Hoppper Medium
python research/finetune_omtm/finetune.py wandb.group="omtm_hopper_m_finetune" pretrain_model_path=../../../outputs/omtm_mae/hopper-medium-v2_140010.pt finetune_args.plan_guidance=critic_lambda_guiding pretrain_args.env_name=hopper-medium-v2 pretrain_dataset.env_name=hopper-medium-v2 finetune_args.seed=1 finetune_args.explore_steps=200000

# ***************************
# Goal Reaching
# ***************************
# D4RL Hoppper Wiggle
python research/zeroshot_omtm/unseen.py --config-name=config_hopper