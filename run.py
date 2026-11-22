import socket
from collections import deque
import struct
import numpy as np
import cv2
import torch
import os
import helpers as h
import ffnn as m
import pg_algo as a

torch.set_num_threads(1) # Makes the state prep operations more efficient

port = 5555

num_frames = 4 # number of frames to pass to model
header_size = 28

render = False

train = False

save = True
load = True

save_steps = 20000 # min number of steps between saves

model_number = "00"
current_level = "stereo_madness"

load_episode = 18255
load_level = "stereo_madness"

model_folder = f"saves/{a.code}_{m.code}{model_number}"
save_folder  = f"{model_folder}/{current_level}"
load_file    = f"{model_folder}/{load_level}/e{load_episode}.pt" if load else None

if save:
    os.makedirs("saves", exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    if not os.path.exists(f"{save_folder}/metrics.csv"):
        with open(f"{save_folder}/metrics.csv", "w") as file:
            file.write(a.header + "\n")

def get_reward(dead, won):
    if dead:
        return -1
    else:
        return 0

step_counter = 0
save_pending = False

frames = deque(maxlen=num_frames)
restart = True

a.init(m.Model, train, load_file, save_folder, current_level == load_level)

if train:
    print("training model...")
else:
    print("running inference...")

print("Waiting for Geometry Dash...")

with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
    sock.setblocking(True)

    print("Connected to Geometry Dash!")

    while True:
        # Receive frame header
        header = h.recv_exact(sock, header_size)

        dead, won, quit, progress, width, height, nbytes = struct.unpack("!7I", header)

        percent = progress / 100.0

        #if restart:
        #    print(f"dead: {dead}, won: {won}, quit: {quit}, percent: {progress/100.0}%, \nwidth: {width}, height: {height}, bytes: {nbytes}")
        if nbytes != width * height:
            raise ValueError("Unexpected pixel format")

        if dead or won or quit:

            if train:
                if a.step_model(percent, get_reward(dead, won), None, dead, won, quit, save_pending):
                    save_pending = False


            if dead: print(f"Player dead at {percent}%. Waiting for respawn...\n")
            if won: print("Level won. Waiting for respawn...\n")
            if quit: 
                print("Level quit.\n")
                break
                    
            restart = True
            continue

        # Check frame metadata
        if width == 0 or height == 0 or not (0 < nbytes <= 50_000_000):
            raise ValueError("Invalid frame metadata")

        # Receive image
        pixels = h.recv_exact(sock, nbytes)

        # pixels is a bytes or bytearray object
        x = torch.frombuffer(pixels, dtype=torch.uint8)
        x = x.reshape(height, width)

        if restart:
            h.reset_buffer(frames, x, num_frames)

            if train:
                action = a.step_model(percent, None, h.get_observation(frames))
            else:
                action = a.infer(h.get_observation(frames))

            restart = False
        else:
            h.add_frame(frames, x)

            if train:
                action = a.step_model(percent, get_reward(dead, won), h.get_observation(frames))
            else:
                action = a.infer(h.get_observation(frames))

        if action == 1:
            sock.sendall(b'1')
            #print("jump!")
        elif action == 0:
            sock.sendall(b'0')
            #print("dont!")
        else:
            print("Invalid action!")
            break

        step_counter += 1
        if save and step_counter % save_steps == 0:
            save_pending = True
        
        if render:
            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape((height, width))

            cv2.imshow("Geometry Dash", frame)
            cv2.waitKey(1)