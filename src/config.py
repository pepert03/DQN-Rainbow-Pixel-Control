import os
import yaml
import torch
import matplotlib

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = "./configs/hyperparameters.yml"


def load_config(hyperparameter_set):
    """Load config from runs folder (if resuming) or from main configs folder."""
    config_file = os.path.join(RUNS_DIR, hyperparameter_set, "config.yml")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(CONFIG, "r") as f:
            all_config = yaml.safe_load(f)
            config = all_config[hyperparameter_set]
            # Save the config to the runs folder for future reference
            os.makedirs(os.path.join(RUNS_DIR, hyperparameter_set), exist_ok=True)
            with open(config_file, "w") as f:
                yaml.dump(config, f)
    return config


def get_paths(hyperparameter_set):
    """Return dict of file paths for a given run."""
    base = os.path.join(RUNS_DIR, hyperparameter_set)
    return {
        "log": os.path.join(base, "training.log"),
        "model": os.path.join(base, "best_model.pt"),
        "checkpoint": os.path.join(base, "checkpoint.pt"),
        "graph": os.path.join(base, "graph.png"),
        "tensorboard": os.path.join(base, "tensorboard"),
    }
