import numpy as np
import torch
from typing import Dict, Optional, Sequence, Tuple, Union


def create_rcbc_mask(traj_length:int, device: str,
                            idx: int) -> Dict[str, torch.Tensor]:
    
    """Predict the action at idx given expected return, current state 
    and history state-action pair"""
    
    state_mask = np.zeros(traj_length)
    state_mask[:idx+1] = 1
    return_mask = np.zeros(traj_length)
    return_mask[idx] = 1
    action_mask = np.zeros(traj_length)
    if idx > 0:
        action_mask[:idx] = 1
    reward_mask = np.zeros(traj_length)
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

def create_fd_mask(traj_length:int, device: str,
                            idx: int) -> Dict[str, torch.Tensor]:
    
    """Predict the state at idx given 
    action-state pair history"""
    state_mask = np.zeros(traj_length)
    state_mask[:idx] = 1
    return_mask = np.zeros(traj_length)
    action_mask = np.ones(traj_length)
    action_mask[idx:] = 0
    reward_mask = np.zeros(traj_length)
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }
    

def create_rew_mask(traj_length:int, device: str,
                            idx: int) -> Dict[str, torch.Tensor]:
    """Predict the reward sequence at idx given 
    action state pair"""
    state_mask = np.zeros(traj_length)
    state_mask[idx] = 1
    action_mask = np.zeros(traj_length)
    action_mask[idx] = 1
    return_mask = np.zeros(traj_length)
    reward_mask = np.zeros(traj_length)
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }
    
    

def create_full_random_mask(
    data_shape: Tuple[int, int],
    traj_length: int,
    mask_ratios: Union[Tuple[float, ...], float],
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    L = traj_length
    P, _ = data_shape

    if isinstance(mask_ratios, Sequence):
        if rnd_state is None:
            mask_ratio = np.random.choice(mask_ratios)
        else:
            mask_ratio = rnd_state.choice(mask_ratios)
    else:
        mask_ratio = mask_ratios

    masked = int(L * P * float(mask_ratio))
    random_mask = np.concatenate(
        [
            np.ones(masked),
            np.zeros(L * P - masked),
        ]
    )
    if rnd_state is None:
        np.random.shuffle(random_mask)
    else:
        rnd_state.shuffle(random_mask)

    random_mask = torch.tensor(random_mask, device=device)
    return random_mask.reshape(L, P)

def create_random_autoregressize_mask(
    data_shapes, mask_ratios, traj_length, device, p_weights=(0, 0, 0.7, 0.3)
) -> Dict[str, np.ndarray]:
    mode_order = ["states", "returns", "actions", "rewards"]
    random_mode = np.random.choice(mode_order, p=p_weights)
    random_position = np.random.randint(0, traj_length)
    masks = {}

    for k, v in data_shapes.items():
        # create a random mask, different mask for each modality
        masks[k] = create_full_random_mask(v, traj_length, mask_ratios, device)

    end_plus_one = False
    for k in mode_order:
        if k == random_mode:
            end_plus_one = True

        # mask out future
        if k in masks:
            if end_plus_one:
                masks[k][random_position:, :] = 0
            else:
                masks[k][random_position + 1 :, :] = 0

    # print(random_mode, random_position)
    return masks

