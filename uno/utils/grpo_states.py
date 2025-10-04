from dataclasses import dataclass
from typing import List
import numpy as np
import random

@dataclass
class GRPOTrainingStates:
    """Parameters for progressive grpo training strategy.
    
    This class manages the parameters and state for progressive training, where
    training is done in groups of timesteps.
    
    Attributes:
        iters_per_group (int): Number of iterations to train on each group of timesteps
        group_size (int): Number of timesteps in each group
        max_timesteps (int): Maximum number of timesteps to train on
        cur_timestep (int): Current timestep being trained on
        cur_iter_in_group (int): Current iteration within the current group
        sample_strategy (str): Strategy for sampling timesteps ("progressive", "random", "decay" or "exp_decay")
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
        max_iters_per_group (int): Maximum number of iterations per group for decay strategy
        min_iters_per_group (int): Minimum number of iterations per group for decay strategy
        roll_back (bool): Whether to roll back to the initial timestep when max_timesteps is reached
        prog_overlap (bool): Whether to overlap the progressive sampling
        prog_overlap_step (int): Step size for progressive overlap
        exp_decay_thre_timestep (int): Threshold timestep for exponential decay strategy
        exp_decay_k (float): Decay rate for exponential decay strategy
    """
    iters_per_group: int
    group_size: int
    max_timesteps: int
    cur_timestep: int = 0
    cur_iter_in_group: int = 0
    sample_strategy: str = "progressive"
    prog_overlap: bool = False
    prog_overlap_step: int = 1
    max_iters_per_group: int = None
    min_iters_per_group: int = None
    roll_back: bool = False
    exp_decay_thre_timestep: int = 13
    exp_decay_k: float = 0.1

    def set_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)
    
    def __post_init__(self):
        if self.sample_strategy == "decay":
            if self.max_iters_per_group is None:
                self.max_iters_per_group = self.iters_per_group
            if self.min_iters_per_group is None:
                self.min_iters_per_group = max(1, self.iters_per_group // 4)
        self.init_timestep = self.cur_timestep
    
    def get_dynamic_iters_per_group(self) -> int:
        """Calculate the dynamic number of iterations per group based on current timestep.
        
        Returns:
            int: Number of iterations for the current group.
        """
        if self.sample_strategy != "decay":
            return self.iters_per_group
            
        # Linear interpolation between max_iters_per_group and min_iters_per_group
        progress = self.cur_timestep / self.max_timesteps
        current_iters = int(self.max_iters_per_group * (1 - progress) + self.min_iters_per_group * progress)
        return max(self.min_iters_per_group, current_iters)
    
    def get_exp_decay_iters_per_group(self) -> int:
        """Calculate the number of iterations per group for exponential decay strategy.
           Formula: y(t) = iters_per_grounp * exp(-k * ReLU(t-threshold)) 
        Returns:
            int: Number of iterations for the current group.
        """
        if self.sample_strategy != "exp_decay":
            return self.iters_per_group
        
        # Exponential decay: double the group size each time
        relu_value = max(0, self.cur_timestep - self.exp_decay_thre_timestep)
        decay_value = self.iters_per_group * np.exp(-self.exp_decay_k * relu_value)
        
        # Ensure the decay value is at least 1
        return np.ceil(decay_value)

    def update_iteration(self, seed = None) -> None:
        """Update the current iteration counter and timestep if needed."""
        if self.sample_strategy == "progressive":
            self.cur_iter_in_group += 1
            if self.cur_iter_in_group >= self.iters_per_group:
                self.cur_iter_in_group = 0
                if self.prog_overlap:
                    self.cur_timestep += self.prog_overlap_step
                else:
                    self.cur_timestep += self.group_size
            if self.cur_timestep + self.group_size - 1 >= self.max_timesteps:
                if self.roll_back:
                    self.roll_back_start()
                else:
                    # 可选，强制clip
                    self.cur_timestep = self.max_timesteps
        elif self.sample_strategy == "random":
            rng = np.random.default_rng(seed)
            self.cur_timestep = rng.integers(0, self.max_timesteps - self.group_size + 1)
        elif self.sample_strategy == "decay":
            self.cur_iter_in_group += 1
            current_iters = self.get_dynamic_iters_per_group()
            if self.cur_iter_in_group >= current_iters:
                self.cur_iter_in_group = 0
                if self.prog_overlap:
                    self.cur_timestep += self.prog_overlap_step
                else:
                    self.cur_timestep += self.group_size
            if self.cur_timestep > self.max_timesteps:
                if self.roll_back: # roll back to the start
                    self.roll_back_start()
                else: # Clip
                    self.cur_timestep = self.max_timesteps
        elif self.sample_strategy == "exp_decay":
            self.cur_iter_in_group += 1
            current_iters = self.get_exp_decay_iters_per_group()
            if self.cur_iter_in_group >= current_iters:
                self.cur_iter_in_group = 0
                if self.prog_overlap:
                    self.cur_timestep += self.prog_overlap_step
                else:
                    self.cur_timestep += self.group_size
            if self.cur_timestep > self.max_timesteps:
                if self.roll_back:
                    self.roll_back_start()
                else:  # Clip
                    self.cur_timestep = self.max_timesteps
        
        else:
            raise ValueError(f"Invalid sample strategy: {self.sample_strategy}")
        

    def roll_back_start(self) -> None:
        """Roll back the current timestep to the init timestep"""
        self.cur_timestep = self.init_timestep
        self.cur_iter_in_group = 0

    def get_current_timesteps(self) -> List[int]:
        """Get the list of timesteps to train on in the current group.
        
        Returns:
            List[int]: List of timesteps to train on. 
            For example, if cur_timestep=5 and group_size=2, returns [5, 6].
        """
        return list(range(self.cur_timestep, min(self.cur_timestep + self.group_size, self.max_timesteps)))

    def is_training_complete(self) -> bool:
        """Check if training is complete.
        
        Returns:
            bool: True if cur_timestep >= max_timesteps, False otherwise.
        """
        if self.sample_strategy in ["progressive", "decay"]:
            return self.cur_timestep >= self.max_timesteps
        
        return False