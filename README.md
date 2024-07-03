### Install python packages from scratch
If you want to make an env from scratch

Make a new conda env
```
conda create -n m3pc python=3.10
conda activate m3pc
```

Install torch with gpu
https://pytorch.org/get-started/locally/


Run these commands to install all dependencies
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -e .
```

Optionally install dev packages.
```
pip install -r requirements_dev.txt
```

### Experiments

Example commands can be found in `train_exsamples.sh`

The main code for offline RL (with online finetuning) and goal reaching is located in the `finetune_omtm` and `zeroshot_omtm`, respectively. Here is how you can run some of the experiments.
 * For pretrain: 
 ```
 python research/omtm/train.py +exp_mtm=d4rl_cont dataset.env_name=hopper-medium-v2 
 ```
 * For offline RL inference:
 ```
 python research/finetune_omtm/finetune.py finetune_args.plan_guidance=rtg_guiding pretrain_args.env_name=hopper-medium-v2 pretrain_dataset.env_name=hopper-medium-v2 finetune_args.explore_steps=0 finetune_args.warmup_steps=0
 ```
 * For online finetuning:
 ```
 python research/finetune_omtm/finetune.py finetune_args.plan_guidance=critic_lambda_guiding pretrain_args.env_name=hopper-medium-v2 pretrain_dataset.env_name=hopper-medium-v2 finetune_args.explore_steps=0 finetune_args.warmup_steps=0
 ```

 * For goal reaching inference:
 ```
 python research/zeroshot_omtm/unseen.py --config-name=config_hopper
 ```

# License & Acknowledgements
This source code is built upon is licensed [facebookresearch/mtm](https://github.com/facebookresearch/mtm) under the MIT license as well as  the following third party dependencies.
 * [ikostrikov/jaxrl](https://github.com/ikostrikov/jaxrl): A fast Jax library for RL. We used this environment wrapping and data loading code for all d4rl experiments.
 * [brentyi/tyro](https://github.com/brentyi/tyro): Argument parsing and configuration
