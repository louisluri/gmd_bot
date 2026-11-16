import socket
from collections import deque
import struct
import numpy as np
import cv2
import torch
import helpers
import pg_ffnn as m
import pg_algo as a

port = 5555

if torch.cuda.is_available():
    device = torch.device(
        "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    )
else:
    device = torch.device("cpu")

num_frames = 4
header_size = 24

render = False

train = True

save = False    # save model and data when done?
model_number = "00"
save_name = f"a_{a.algo_code}_m_{m.model_code}{model_number}"

load = False
load_step = 6867
load_file = f"checkpoints/{save_name}_step{load_step}.pt" if load else None

frames = deque(maxlen=num_frames)

def get_reward(dead, won):
    if dead:
        return -10
    elif won:
        return 10
    else:
        return 0

i = 0
restart = True
try:
    i = a.init(m.Model, load_file, device, train)

    if train:
        print("training model...")
    else:
        print("running inference...")

    print("Waiting for Geometry Dash...")

    with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
        sock.settimeout(60.0)

        print("Connected to Geometry Dash!")

        while True:
            # Receive frame header
            header = helpers.recv_exact(sock, header_size)

            dead, won, quit, width, height, nbytes = struct.unpack("!IIIIII", header)

            if i == 0:
                print(f"dead: {dead}, won: {won}, quit: {quit}, \nwidth: {width}, height: {height}, bytes: {nbytes}")
            if nbytes != width * height:
                raise ValueError("Unexpected pixel format")

            if dead or won or quit:
                if train:
                    a.step(device=device, step=i, reward=get_reward(dead, won), state=None, 
                        dead=dead, won=won, quit=quit)

                if dead: print("Player dead. Waiting for respawn...")
                if won: print("Level won. Waiting for respawn...")
                if quit: 
                    print("Level quit.")
                    break
                        
                restart = True
                continue

            # Check frame metadata
            if width == 0 or height == 0 or not (0 < nbytes <= 50_000_000):
                raise ValueError("Invalid frame metadata")

            # Receive image
            pixels = helpers.recv_exact(sock, nbytes)

            # pixels is a bytes or bytearray object
            x = torch.frombuffer(pixels, dtype=torch.uint8)
            x = x.reshape(height, width)

            if restart:
                helpers.reset_buffer(frames, x, num_frames)

                if train:
                    action = a.step(device, i, None, helpers.get_observation(frames))
                else:
                    action = a.infer(device, helpers.get_observation(frames))
                restart = False
            else:
                helpers.add_frame(frames, x)

                if train:
                    action = a.step(device, i, get_reward(dead, won), helpers.get_observation(frames))
                else:
                    action = a.infer(device, helpers.get_observation(frames))

            i += 1

            if action == 1:
                sock.sendall(b'1')
            elif action == 0:
                sock.sendall(b'0')
            else:
                print("Invalid action!")
                break
            

            if render:
                frame = np.frombuffer(pixels, dtype=np.uint8)
                frame = frame.reshape((height, width))

                cv2.imshow("Geometry Dash", frame)
                cv2.waitKey(1)

except (OSError, ValueError, KeyboardInterrupt) as e:
    print(f"Error encountered: {e}")
finally:
    print("finally")
    if save and a.last_sucessful != 0 and train:
        print("Saving and exiting...")
        helpers.save_checkpoint(a.last_sucessful, a.episode, save_name, a.header, a.m, a.o, a.history)