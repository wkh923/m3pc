### Install python packages from scratch
If you want to make an env from scratch

Make a new conda env
```
conda create -n mtm python=3.10
conda activate mtm
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

refer to `train_exsamples.sh`

# License & Acknowledgements
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree. This is not an official Meta product.

This project builds on top of or utilizes the following third party dependencies.
 * [FangchenLiu/MaskDP_public](https://github.com/FangchenLiu/MaskDP_public): Masked Decision Prediction, which this work builds upon
 * [ikostrikov/jaxrl](https://github.com/ikostrikov/jaxrl): A fast Jax library for RL. We used this environment wrapping and data loading code for all d4rl experiments.
 * [brentyi/tyro](https://github.com/brentyi/tyro): Argument parsing and configuration
