import socket
from collections import deque
import struct
import time
import numpy as np
import cv2
import torch
import helpers
import pg_model as m
import pg_algo as a

host = "127.0.0.1"
port = 5555

header_size = 24

render = False

save = True    # save model and data when done?
model_number = "00"
save_name = f"a{a.algo_code}_m{m.model_code}{model_number}"

load = False
load_step = 100
load_file = f"checkpoints/{save_name}_step{load_step}.pt" if load else None

frames = deque(maxlen=4)

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
    i = a.init(m.Model, load_file)

    print("Waiting for Geometry Dash...")

    with socket.create_connection((host, port), timeout=3) as sock:
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
                a.step(step=i, reward=get_reward(dead, won), state=None, 
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
            if (width, height) != (m.width, m.height):
                raise RuntimeError(f"Model expects {m.width}x{m.height}, received {width}x{height}")

            # Receive image
            pixels = helpers.recv_exact(sock, nbytes)

            # pixels is a bytes or bytearray object
            x = torch.frombuffer(pixels, dtype=torch.uint8)
            x = x.reshape(height, width)

            if restart:
                helpers.reset_buffer(frames, x)

                action = a.step(i, None, helpers.get_observation(frames))
                restart = False
            else:
                helpers.add_frame(frames, x)

                action = a.step(i, get_reward(dead, won), helpers.get_observation(frames))

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
    print("Python dissconected")
    print(f"Error encountered: {e}")
finally:
    if save and a.last_sucessful != 0:
        print("Saving and exiting...")
        helpers.save_checkpoint(a.last_sucessful, a.episode, save_name, a.header, a.m, a.o, a.history)