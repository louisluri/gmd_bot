# Geometry Dash RL Model

A Geode mod and Python script for controlling Geometry Dash
frame by frame to train a reinforcement learning agent.


## Plan

People online have been rather unsuccessfull training RL agents to play Geometry Dash. My plan is to test multiple strategies aginst eachother. Ultimately i think using a convolutional network along with a transformer to grasp the relationship between feaures will perform better. I am not sure which RL algorithm will best suit that model yet so i'll test a few


## Layout

"rl_env" folder is a geode mod, the "src" subfolder holds the actual code I wrote. Everything else just was there when I initialized the mod. "mod.json" shows the properties.

"run.py" is the python script you run that interacts with Geometry Dash and trains the agent or runs inference.

There will be diffrent "algo" files and model files to train with diffrent RL algorithms and use them on diffrent types of models. So just import them into run.py instead of whatever is currently imported to switch them. (some algos will only work with certain models)


## Requirements

- Geometry Dash 2.2081
- Geode 5.11.2
- torch


## How It Works

The Geode mod connects to Python through a socket.
GD sends a frame scaled down by a scaling factor and in greyscale along with a boolean flag representing if the player died, won or quit.
Python sends a 1 or 0 back to advance the game and control the player.