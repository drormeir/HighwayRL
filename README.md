# HighwayRL
This project was submitted as part of [Reichman University Reinforcement Learning graduate course 3640](https://www.runi.ac.il/)

The purpose of the project is to build and train a single agent that navigates a car in the highway environment, trying to reach optimal speed and avoiding any collisions with other cars in a simulation of a highway.

This environment is episodic, where each episode is consists of 500 steps at most. The environment provides a reward for each step which can vary between 0.0 to 1.0.
The goal of the agent is to maximize the total reward of an episode by driving as fast as possible and avoiding crashes with other cars.

The agent runs on Python 3.8 + PyTorch. The paper that describes the algorithm is ["Double Duel Q-network"](https://arxiv.org/abs/1511.06581) with "Epsilon-Greedy policy" for environment exploration and an "Experience Replay Buffer" as a dynamic dataset for the learning process.


# Installation
Use the Colab notebook "RL_Final_Project_2022.ipynb" or install the python libraries according to "requirements.txt"

# Report
A detailed report describing the learning algorithm, along with ideas for future work is at [report.md](https://github.com/drormeir/HighwayRL/blob/master/Report.md)
