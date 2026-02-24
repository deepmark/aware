from math import tau

import yaml
import torch
import numpy as np

def load_config(config_path: str) -> dict:
    """
    Load a config file from a given path
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config from {config_path}: {e}")
    
def to_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert a numpy array or torch tensor to a torch tensor
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        return data
    
def BRH_activation_to_probability(average_activation: float, agreement: float, mode_name: str) -> float:
    
    if mode_name == "full_length":
        x0 = 0.041
        k_L = 137.3265360835137    
        k_R = 91.55102405567581    

        x = np.asarray(average_activation)
        left = 1.0 / (1.0 + np.exp(-k_L * (x - x0)))
        right = 1.0 / (1.0 + np.exp(-k_R * (x - x0)))
        out = np.where(x < x0, left, right)

        tau = float(out)
        
        return tau

    #=================================================================================================
    x0 = 0.065
    k_L = np.log(9.0) / 0.035      
    k_R = np.log(9.0) / 0.015 

    x = np.asarray(average_activation)
    left = 1.0 / (1.0 + np.exp(-k_L * (x - x0)))
    right = 1.0 / (1.0 + np.exp(-k_R * (x - x0)))
    out = np.where(x < x0, left, right)

    tau1 = float(out)
    if tau1 >= 0.5:
        return tau1

    x0 = 0.55
    k_L = np.log(9.0) / 0.15      
    k_R = np.log(9.0) / 0.35      

    x = np.asarray(agreement)
    left = 1.0 / (1.0 + np.exp(-k_L * (x - x0)))
    right = 1.0 / (1.0 + np.exp(-k_R * (x - x0)))
    out = np.where(x < x0, left, right)

    tau2 = float(out)

    if tau2 >= 0.5:
        return tau2
    
    return tau1