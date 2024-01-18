"""
In this file I will include utils for argument parsers.
"""
import os
import argparse



def check_positive_float(num):
    num_float = float(num)

    if not (num_float > 0.0):
        raise argparse.ArgumentTypeError(f"\'{num_float}\' is not a positive float")

    return num_float



def check_positive_int(num):
    num_int = int(num)

    if not (num_int > 0.0):
        raise argparse.ArgumentTypeError(f"\'{num_int}\' is not a positive integer")

    return num_int



def check_dir_exists(dir_path):
    dir_path_str = str(dir_path)

    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(f"Directory \'{dir_path_str}\' does not exist.")

    return dir_path_str



def check_target_model_type(model_type):
    model_type_str = str(model_type).lower()

    if model_type_str not in ['sgt', 'gbr', 'xgboost']:
        raise argparse.ArgumentTypeError("Target model type should be equal to sgt, gbr or xgboost(non case-sensitive)")
    
    return model_type_str



def check_target_model_sampling_strategy(strategy):
    strategy_str = str(strategy)

    if strategy_str not in ['recent', 'recent_window', 'exp_gain', 'all']:
        raise argparse.ArgumentTypeError("Target Model\'s sampling strategy should be equal to recent, recent_window, exp_gain or all")
    
    return strategy_str