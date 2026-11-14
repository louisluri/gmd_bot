# Geometry Dash RL Model

A Geode mod and Python script for controlling Geometry Dash
frame by frame to train a reinforcement learning agent.


## Plan

People online have been rather unsuccessfull training RL agents to play Geometry Dash. My plan is to use a convolutional network along with a transformer to grasp the relationship between feaures better. I am not sure which RL algorithm will best suit that yet so I'll
test a few


## Layout

"rl_env" folder is a geode mod, the "src" subfolder holds the actual code I wrote. Everything else just was there when I initialized the mod. "mod.json" shows the properties.

"train.py" is the python script you run that interacts with Geometry Dash and trains the agent.


## Requirements

- Geometry Dash 2.2081
- Geode 5.11.2
- torch


## How It Works

The Geode mod connects to Python through a socket.
GD sends a frame scaled down by 1/4 and in greyscale along with a boolean flag representing if the player died.
Python sends a 1 or 0 back to advance the game and control the player.