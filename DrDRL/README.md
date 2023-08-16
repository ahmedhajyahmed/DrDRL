# Dr. DRL
Dr. DRL : a Deep Reinforcement Learning approach for adapting  agent in drifted environments.
> Dr. DRL leverage intentional forgetting to reset hypoactive neurons with low-scaled weights.
> This erases the agent’s minor behaviors prior to fine-tuning in the drifted environment.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Getting started](#getting-started)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)


## General Information
- Deep Reinforcement Learning (DRL)
- Enhancement of DRL Plasticity
- Intentional Learning
- Dual Speed Learning

## Technologies Used
- python 3.7.7
- TensorFlow 2.4.4

## Installation
For installation, we recommend using anaconda and creating a new virtual environment to test Diverget. Steps to install 
using Anaconda will be as follows :
- Install Anaconda [_here_](https://www.anaconda.com/products/individual)
- Create a new virtual environment : `conda create -n DrDRL_demo python=3.7.7`
- Activate this environment : `conda activate DrDRL_demo`
- Go to DrDRL Directory
- Install requirements : `pip install -r requirements.txt`
- Install our package:
  - `pip install -e .`
- You're good to go now !!!

## Code Structure
Our code contain two main directories (packages):
- dr_drl: our core framework.
- experiments: a directory containing example scripts for reproduction.


## Getting started
In this section, we will try to go all the way through how to use Dr. DRL. We will work with this configuration:
- DRL agent : DQN
- Environment : CartePole

You can follow the next steps to work with any configuration you want.

###Training 
First we need to train the DRL agent on the original Environment. In our case we are going to train DQN with CartePole.
For that 
- `cd [your path]/experiments/cartepole/dqn`
- `python dqn_cartepole_train.py`

###Comparing Dr. DRL to Continual Learning 
To Comparing Dr. DRL to Continual Learning we need to:

- `cd [your path]/experiments/cartepole/dqn`
- `python dqn_cartepole_repair.py`

The process is quite slow. The script will generate 6 environmental drift for each DRL agent instance.

## Project Status
Project is: _in progress_.



  
