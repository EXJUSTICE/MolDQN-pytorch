# MolDQN-pytorch
[![MIT
license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

PyTorch implementation of MolDQN as described in [Optimization of Molecules via Deep Reinforcement Learning](https://www.nature.com/articles/s41598-019-47148-x)
by Zhenpeng Zhou, Steven Kearnes, Li Li, Richard N. Zare and Patrick Riley.

<p align="center">
  <img src= "https://github.com/EXJUSTICE/MolDQN-pytorch/blob/master/MOLdqnPAPER.jpg?raw=true">
</p>

## Background
TBD
## Contents
*  main.py<br/>
   \n Primary training script. Initializes the training environment, the agent, and performs training for n interations. Hyperparameters    can be found in hyp.py (Coming Soon: Argparse arguments)
*  environment.py<br/> Base MDP environment class. Defines the chemical methods and markov decision process for molecular generation
*  dqp.py<br/> Model architecture for the agent.
*  agent.py<br/> Base agent class. Defines all methods available to the agent with regards to action selection, replay storage, reward processing, and parameter updates.
*  molecules.py<br/> Auxilliary classes defining the LogP loss function, and several Rdkit computations

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
   
## From Docker:

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
## Training the MolDQN:

`python main.py`

A Notebook to train the model on a single property QED optimization task can be seen in `examples/MolDQN-pytorch.ipynb`.
Another Notebook to train the model on a single property LogP optimization task can be seen in `examples/MolDQN_LogP.ipynb`.

Note that the example notebooks can be run over online GPU instances such as Google Colaboratory - however this is not recommended due to extreme training time.

## Results:

The following was the reward curve obtained when the model was trained for 5000 episodes on a single property optimization task (QED in this case).

<img src="https://github.com/aksub99/MolDQN-pytorch/blob/master/Results/plots/episode_reward.svg" height="500" width="500">
