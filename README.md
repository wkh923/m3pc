# M^3PC: Test-Time Model Predictive Control for Pretrained Masked Trajectory Model
This repo is for the paper [`M^3PC: Test-Time Model Predictive Control for Pretrained Masked Trajectory Model`](https://arxiv.org/abs/2412.05675). The code is built upon [facebookresearch/mtm](https://github.com/facebookresearch/mtm).

## Overview
![M3PC](https://i.postimg.cc/T1sgpqd9/3way-ellip.png)

**M^3PC** is an approach that leverages properly designed masking schemes to perform test-time MPC with pretrained masked trajectory models for decision making tasks. It enables action reconstruction with uncertainties for better robustness, as well as forward and backward prediction through different masking patterns for solving various downstream tasks. See detailed explanation and demo video  [here](https://drive.google.com/file/d/1d6gVwZK650SoQpsFqzc61LtQBlk0-p7Y/view?usp=sharing)

## Usage

### Install python packages from scratch
If you want to make an env from scratch

Make a new conda env
```
conda create -n m3pc python=3.10
conda activate m3pc
```


Run these commands to install all dependencies
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
```
pip install -e .
```
### Troubleshooting

For torch import issue, update MKL
```
conda install mkl==2024.0
```

For gym install issue, downgrade pip
```
pip install setuptools==65.5.0 pip==21 
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
 You can also directly download the pretrain model checkpoints [here](https://polybox.ethz.ch/index.php/s/UBaK1WwziIpxl4d) and extract the file in the root directory:
 ```
 tar -xf checkpoints.tar 
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
change `--config-name` and select files from `waypoint_gen` folder for different tasks and different goals.
You can also try modify the waypoint generation [script](https://github.com/wkh923/m3pc/blob/main/research/zeroshot_omtm/waypoint_gen/gen_and_vis.py) to control the wiggling frequency, visualizaing the waypoints and see to what extent the backward M3PC can follow 'unseen' state trajectory.

## Citation
If you find M^3PC useful in your research or if you refer to the results mentioned in our work, please star this repository and consider citing:

```bibtex
@article{M3PC,
  title={M^3PC: Test-Time Model Predictive Control for Pretrained Masked Trajectory Model},
  author={Kehan Wen and Yutong Hu and Yao Mu and Lei Ke},
  journal={arxiv:2412.05675},
  year={2024},
}
```

# License & Acknowledgements
This work is licensed under BSD 3-Clause License. See [LICENSE](LICENSE) for details. Third-party datasets and tools are subject to their respective licenses.

