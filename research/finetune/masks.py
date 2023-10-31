import numpy as np
import torch
from typing import Dict


def create_rcbc_mask(traj_length:int, device: str,
                            pos: int) -> Dict[str, torch.Tensor]:
    
    """Predict the action sequence from pos to end given expected return and history state-action pair"""
    
    state_mask = np.zeros(traj_length)
    state_mask[:pos+1] = 1
    return_mask = np.ones(traj_length)
    action_mask = np.zeros(traj_length)
    if pos > 0:
        action_mask[:pos] = 1
    reward_mask = np.zeros(traj_length)
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

def create_fd_mask(traj_length:int, device: str,
                            pos: int) -> Dict[str, torch.Tensor]:
    
    """Predict the state-reward sequence from pos to end given 
    all actions and history state-reward pair"""
    state_mask = np.zeros(traj_length)
    state_mask[:pos+1] = 1
    return_mask = np.zeros(traj_length)
    action_mask = np.ones(traj_length)
    reward_mask = np.zeros(traj_length)
    if pos > 0:
        reward_mask[:pos] = 1
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }
    

def create_random_mask(traj_length: int, device: str,
                        mask_type: str) -> Dict[str, torch.Tensor]:
    """Make future sequence prediction based on current RTG, state and action"""

    if mask_type not in ["rcbc", "fd"]:
        raise ValueError("mask type error")

    pos = np.random.choice(traj_length)
    
    # Dynamically call the appropriate mask creation function
    mask_function = globals().get(f'create_{mask_type}_mask')
    
    if mask_function:
        return mask_function(traj_length, device, pos)
    else:
        raise NameError(f"No function found for mask type: {mask_type}")

