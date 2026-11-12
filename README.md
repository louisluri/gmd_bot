# Geometry Dash RL Model

A Geode mod and Python script for controlling Geometry Dash
frame by frame to train a reinforcement learning agent.


## Plan

People online have been rather unsuccessfull training RL agents to play Geometry Dash. My plan is to use a convolutional network along with a transformer to grasp the relationship between feaures better. I am not sure which RL algorithm will best suit that yet so I'll
test a few


## Features

- Advance the game through Python commands
- Control jump input
- Render game frames in Python


## Requirements

- Geometry Dash 2.2081
- Geode 5.11.2
- torch


## Setup

1. Clone this repository.
2. Download and set up Geode
3. While in the mod folder run: geode build


## Usage

1. Open Geometry Dash and set your fps to the same as your step_dt in the mod.
2. Go to geode settings and activate the terminal so that you can see if errors arise
3. Open a level, the game should freeze on the first step
4. Run the Python train script


## How It Works

The Geode mod connects to Python through a socket.
GD sends a frame scaled down by 1/4 and in greyscale along with a boolean flag representing if the player died.
Python sends a 1 or 0 back to advance the game and control the player.