import numpy as np
import torch
from typing import Dict


def create_rcbc_mask(traj_length: int, device: str
                     ) -> Dict[str, torch.tensor]:
    """Return based policy: output an action based on expected return and current state """
    
    
    state_mask = np.zeros(traj_length)
    state_mask[0] = 1
    return_mask = np.zeros(traj_length)
    return_mask[0] = 1
    action_mask = np.ones(traj_length)
    reward_mask = np.ones(traj_length)
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

def create_future_prediction_mask(traj_length: int, device: str
                                   ) -> Dict[str, torch.tensor]:
    
    """Make future sequence prediction based on current RTG, state and action"""
    
    
    state_mask = np.zeros(traj_length)
    state_mask[0] = 1
    return_mask = np.zeros(traj_length)
    return_mask[0] = 1
    action_mask = np.zeros(traj_length)
    action_mask[0] = 1
    reward_mask = np.zeros(traj_length)

    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

