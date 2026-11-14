import torch
import numpy as np
import os

def reset_buffer(frames, first_frame):
    frames.clear()

    # Fill the initial history with the starting frame
    for _ in range(4):
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

def save_checkpoint(step, episode, name, header, model, optim, data):
    os.makedirs("checkpoints", exist_ok=True)
    # Save only the weights (state_dict) to a .pt file
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "step": step,
        "episode": episode
    }
    torch.save(checkpoint, f"checkpoints/{name}_step{step}.pt")

    print("Model checkpoint saved successfully!")

    os.makedirs("metrics", exist_ok=True)
    # Save to CSV, specifying a comma delimiter and a clean header
    np.savetxt(
        f"metrics/{name}_step{step}.csv", 
        np.array(data), 
        delimiter=",", 
        header=header, 
        comments='' # Prevents NumPy from adding a '#' before the header
    )
    print(f"Metrics saved to {name}_step{step}.csv!")