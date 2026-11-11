import socket
import struct
import time
import numpy as np
import cv2
#import model

host = "127.0.0.1"
port = 5555

header_size = 16

render = True

def recv_exact(sock, n):
    data = bytearray()

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError("Disconnected from Geometry Dash")

        data.extend(chunk)

    return data

i = 0
while True:
    try:
        print("Waiting for Geometry Dash...")

        with socket.create_connection((host, port), timeout=3) as sock:
            sock.setblocking(True)

            print("Connected to Geometry Dash!")

            while True:
                # Receive frame header
                header = recv_exact(sock, header_size)

                dead, width, height, nbytes = struct.unpack("!IIII", header)

                if i % 1000 == 0:
                    print(f"dead: {dead}, width: {width}, height: {height}, bytes: {nbytes}")

                # Player died
                if dead:
                    print("Player died. Waiting for respawn...")
                    continue

                # Check frame metadata
                if width == 0 or height == 0 or not (0 < nbytes <= 50_000_000):
                    raise ValueError("Invalid frame metadata")

                # Receive image
                pixels = recv_exact(sock, nbytes)

                if nbytes != width * height:
                    raise ValueError("Unexpected pixel format")

                if render:
                    frame = np.frombuffer(pixels, dtype=np.uint8)
                    frame = frame.reshape((height, width))

                    cv2.imshow("Geometry Dash", frame)
                    cv2.waitKey(1)

                # Send action to Geometry Dash
                if i % 2 == 0:
                    sock.sendall(b'0')
                else:
                    sock.sendall(b'1')

                i += 1

    except (OSError, ValueError) as e:
        print(f"Connection lost: {e}")
        print("Retrying in 1 second...")
        time.sleep(1)