import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:1")
print(device)

model_number = "00"
checkpoint_path = f'checkpoints/model{model_number}.pt'
log_file = f'train_data/model{model_number}_data.csv'
save = False    # save model and data?

# HYPERPARAMS
batch_size = 128
learning_rate = 1e-3

frames_per_observation = 4
space_between_frames = 4
image_width = 512
image_height = 320

n_embed = 256