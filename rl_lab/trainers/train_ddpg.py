import os
from datetime import datetime
import torch
import torch.nn as nn
from ..agents.ddpg import train

def create_folder(base_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"{base_name}_{timestamp}"

    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Get absolute path
    abs_path = os.path.abspath(folder_name)

    print(f"Folder created (or already exists) at: {abs_path}")
    return abs_path


if __name__ == "__main__":

    linear_model = nn.Sequential(
        nn.Linear(3600, 64),  # Layer 1
        nn.ReLU(),
        nn.Linear(64, 32),    # Layer 2
        nn.ReLU(),
        nn.Linear(32, 3600),  # Layer 3
        nn.Sigmoid()
    )

    conv_model = nn.Sequential(
        nn.Unflatten(1, (1, 60, 60)),                # [B, 3600] -> [B, 1, 60, 60]
        nn.Conv2d(1, 64, kernel_size=5, padding=2),  # [B, 64, 60, 60]
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 60, 60]
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),# [B, 64, 60, 60]
        nn.ReLU(),
        nn.Conv2d(64, 1, kernel_size=3, padding=1),  # [B, 1, 60, 60]
        nn.Sigmoid(),
        nn.Flatten(start_dim=1)                      # [B, 1, 60, 60] -> [B, 3600]
    )

    nets = [('conv', conv_model), ('linear', linear_model)]
    episode_lens = [50, 50, 50, 50]

    total_steps = 8000
    for net in nets:
        for ep_len in episode_lens:
            total_episodes = total_steps // ep_len
            model_type, model = net
            experiment_name = f"{model_type}_{ep_len}"
            folder_name = experiment_name + "_dir_"
            folder = create_folder(folder_name)
            print(f"Beginning experiment {experiment_name} to be saveed in {folder_name}.")
            train(total_episodes, ep_len, folder, True, net[1], experiment_name=experiment_name)
            print(f"Experiment {experiment_name} complete.")


