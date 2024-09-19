mamba activate mtm
python research/omtm/train.py +exp_mtm=d4rl_cont dataset.env_name=hopper-medium-replay-v2 local_cuda_rank=0 args.mask_patterns=["SRR"]

mamba activate mtm
python research/omtm/train.py +exp_mtm=d4rl_cont dataset.env_name=walker2d-medium-v2 local_cuda_rank=1 args.mask_patterns=["SRR"]

mamba activate mtm
python research/omtm/train.py +exp_mtm=d4rl_cont dataset.env_name=walker2d-medium-replay-v2 local_cuda_rank=1 args.mask_patterns=["SRR"]

mamba activate mtm
python research/omtm/train.py +exp_mtm=d4rl_cont dataset.env_name=halfcheetah-medium-replay-v2 local_cuda_rank=2 args.mask_patterns=["SRR"]

mamba activate mtm
python research/omtm/train.py +exp_mtm=d4rl_cont dataset.env_name=halfcheetah-medium-v2 local_cuda_rank=2 args.mask_patterns=["SRR"]