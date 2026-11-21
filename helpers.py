import torch
import numpy as np
import os
import csv

def reset_buffer(frames, first_frame, num_frames):
    frames.clear()

    # Fill the initial history with the starting frame
    for _ in range(num_frames):
        frames.append(first_frame.clone().unsqueeze(0))

def add_frame(frames, frame):
    frames.append(frame.clone().unsqueeze(0))

def get_observation(frames):
    # Each frame: (C, H, W)
    # Combine frames along the channel dimension
    stacked = torch.cat(list(frames), dim=0)

    # Add batch dimension
    return stacked.unsqueeze(0)

def recv_exact(sock, n):
    data = bytearray()

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError("Disconnected from Geometry Dash")

        data.extend(chunk)

    return data

def save_checkpoint(model, optim, step, episode, optim_step, history, folder):
    # Save only the weights (state_dict) to a .pt file
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "step": step,
        "episode": episode,
        "optim_step": optim_step
    }
    torch.save(checkpoint, f"{folder}/e{episode}.pt")

    print("Model checkpoint saved successfully!")

    with open(f"{folder}/metrics.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(history)

    print(f"Metrics saved")