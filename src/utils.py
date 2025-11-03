"""
A module for consolidating the utility functions that is used throughout the whole project.

This module defines the set_seed, get_parameter_summary,
setup_logging, get_gflops, and load_config functions for general use.
"""

import torch
import yaml
from torch import nn
import numpy as np
import random
import os
import logging
from datetime import datetime
import pathlib
import sys
from torch.utils.flop_counter import FlopCounterMode

def set_seed(seed: int):
    """
    Sets seeds across all relevant libraries to ensure reproducible results.

    This function sets seeds for `random`, `numpy`, `torch` (CPU and CUDA),
    and configures cuDNN for deterministic behavior.

    Args:
        seed: The integer seed value to use for all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")

def get_parameter_summary(model: nn.Module) -> dict:
    """
    Calculates the total and trainable parameters of a model.
    Args:
        model: The model to get the parameter summary of

    Returns:
        dict: A dictionary containing keys: 'total_params', 'trainable_params', and 'trainable_percentage'.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
    }

def setup_logging(log_dir: pathlib.Path):
    """
    Configures the logging for the training run.
    This function removes all existing handlers, creates a unique log file
    in the specified directory, and adds both a file handler and a
    console handler.
    Args:
        log_dir: path to the directory where the log file is to be written.

    Returns:
        path to the log file
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = log_dir / f"training_run_{timestamp}.log"

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove all existing handlers (e.g., from timm)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Create and add file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create and add console handler (so you see logs live)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_filepath

def get_gflops(model: nn.Module, dummy_input: torch.Tensor) -> float:
    """
    Calculates the total GFLOPs for a model's forward pass using the native counter.
    Args:
        model: The model to get GFLOPs for.
        dummy_input: A sample input tensor to trace the model's forward pass.

    Returns:
        The total GFLOPs
    """
    flop_counter = FlopCounterMode(model)
    with flop_counter:
        model(dummy_input)

    total_flops = flop_counter.get_total_flops()
    return total_flops / 1e9


def load_config(config_path: pathlib.Path) -> dict:
    """
    Loads a YAML config file, handling _base_ inheritance recursively.
    Args:
        config_path: path to the config file.

    Returns:
        The combined config file to be returned
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # If a base config is specified, load it recursively first
    if '_base_' in config:
        # Remove key after reading to prevent it from being in the final config
        base_path_str = config.pop('_base_')
        base_path = config_path.parent / base_path_str

        # Recursive call to traverse through the config hierarchy.
        base_config = load_config(base_path)

        # Deep merge logic
        def deep_merge(base: dict, override: dict) -> dict:
            """
            Recursively merges the 'override' dict into the 'base' dict.

            If a key exists in both and both values are dicts, it recurses.
            Otherwise, the value from 'override' replaces the value from 'base'.

            Args:
                base: The base dictionary to merge into.
                override: The dictionary with new values to merge.

            Returns:
                The merged dictionary (the modified 'base' dict).
            """
            for key, value in override.items():
                if isinstance(value, dict) and key in base and isinstance(base.get(key), dict):
                    base[key] = deep_merge(base.get(key, {}), value)
                else:
                    base[key] = value
            return base

        # Merge the current config on top of the fully resolved base config
        return deep_merge(base_config, config)
    else:
        # This is the root config file (e.g., base_config.yaml)
        return config