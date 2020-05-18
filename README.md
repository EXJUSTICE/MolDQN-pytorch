# MolDQN-pytorch
[![MIT
license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

PyTorch implementation of MolDQN as described in [Optimization of Molecules via Deep Reinforcement Learning](https://www.nature.com/articles/s41598-019-47148-x)
by Zhenpeng Zhou, Steven Kearnes, Li Li, Richard N. Zare and Patrick Riley.

<p align="center">
  <img src= "https://github.com/EXJUSTICE/MolDQN-pytorch/blob/master/utils/moldqn_paper.jpg?raw=true">
</p>

## Background: Q-learning

The MOlDQN paper currently relies solely on Q-learning. Q-learning is a reinforcement learning algorithm belonging to the temporal difference (TD) family of algorithms. Unlike tradtional Monte Carlo approaches, TD utilizes state-value estimates of the next state obtained intra-episode in order to improve its estimate of the state-value of the current state.<br/><br/> Q-learning further builds upon this by creating the estimate of the state-value of the next state using the action that maximizes its value at that time, instead of an expected value (as observed in other TD approaches such as SARSA). A detailed consideration of Q-learning can be found [here](https://towardsdatascience.com/automating-pac-man-with-deep-q-learning-an-implementation-in-tensorflow-ca08e9891d9c).

## Contents
*  `main.py`<br/>Primary training script for single property optimization. Initializes the training environment, the agent, and performs training for n interations. Hyperparameters can be found in `hyp.py`. Capable of optimizing the LogP or QED of a molecule.
*  `main_multi.py`<br/>Primary training script for multi property optimization. Initializes the training environment, the agent, and performs training for n interations. Hyperparameters can be found in `hypmulti.py`. Currently co-optimizes Tanimoto Similarity and QED of a molecule. LogP integration underway.
*  `environment.py`<br/> Base MDP environment class. Defines the chemical methods and markov decision process for molecular generation
*  `dqn.py`<br/> Model architecture for the agent.
*  `agent.py`<br/> Base agent class. Defines all methods available to the agent with regards to action selection, replay storage, reward processing, and parameter updates.
*  `molecules.py`<br/> Auxilliary classes defining the LogP loss function, and several RDkit computations.
*  `examples/`<br/> Folder containing Jupyter Notebook Examples. Both QED and LogP loss functions, as well as multi-objective optimization are demonstrated.

## Installation

## <a name="source"></a>From source:

1) Install `rdkit`.  
   `conda create -c rdkit -n my-rdkit-env rdkit`  
   `conda activate my-rdkit-env`  
   `conda install -c conda-forge rdkit`  
   
2) Clone this repository.  
   `git clone https://github.com/aksub99/MolDQN-pytorch.git`  
   `cd MolDQN-pytorch`
   
3) Install the requirements given in `requirements.txt`.  
   `pip install -r requirements.txt`  
   
4) Install `baselines`.  
   `pip install "git+https://github.com/openai/baselines.git@master#egg=baselines-0.1.6"`  
   
## From Docker

Using a docker image requires an NVIDIA GPU.  If you do not have a GPU please follow the directions for [installing from source](#source)
In order to get GPU support you will have to use the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin.
``` bash
# Build the Dockerfile in Dockerfiles/Dockerfile to create a Docker image.
cd Dockerfiles
docker build -t moldqn_pytorch:latest .

# This will create a container from the image we just created.
nvidia-docker run -[Options] moldqn_pytorch:latest python path/to/main.py
```
Please remember to modify the `TB_LOG_PATH` variable in `main.py` depending on where you wish to store your tensorboard runs file.
## Training the MolDQN

The notebook to train the model on a single property QED optimization task can be seen in `examples/MolDQN-pytorch.ipynb`.
Similarly, the notebook to train the model on a single property LogP optimization task can be seen in `examples/MolDQN_LogP.ipynb`. Hyperparameters used during training can be found in `hyp.py`

Finally, the notebook for multiple property optimization task can be seen in `examples/MolDQN_MultiObjective.ipynb` .Hyperparameters used during training can be found in `hypmulti.py`

Note that the example notebooks can be run over online GPU instances such as Google Colaboratory - however this is not recommended due to extreme training time.

## Results

The following was the reward curve for single property optimization task (QED) obtained when the model was trained for 5000 episodes is shown below:

<img src="https://github.com/aksub99/MolDQN-pytorch/blob/master/Results/plots/episode_reward.svg" height="500" width="500">

The reward curve for the multi-objective optimization (Tanimoto Similarity + QED) obtained when the model was trained for 375 episodes is shown below:

<img src="https://github.com/EXJUSTICE/MolDQN-pytorch/blob/master/Results/plots/episode_reward_multi.PNG?raw=true" >
## TODO

*  Display rewards
*  Output SMILES
*  Implement SARSA, Expected SARSA etc.
*  Evaluate auxilliary RDKIT reward functions

