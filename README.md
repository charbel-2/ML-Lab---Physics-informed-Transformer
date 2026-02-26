# A Physics-Informed In-Context Learning Framework for Online Force Prediction in Contact Tasks

This repo includes initial Python codes of a purely data-driven transformer for online prediction of interaction forces, to be augmented with Physics-informed layers to improve generalisation in Out-of-distribution (OOD) scenarios. The proposed approach extends transformer-based meta learning with physically grounded inductive biases, where students are encouraged to look for and implement one or more modifications to the transformer architecture so it can outperform its data-driven counterpart in OOD scenarios. Those modifications can include, but are not limited to:
  - **Physics-based loss function**,
  - **Physics-aware embedding**,
  - **Physics-focused attention mechanisms**,
  - **Learnable physics parameters**

An example of the resulting architecture is shown in the attached figure, where inputs/outputs of the system are divided as follows:

**Inputs**:


  - **Cartesian Positions**: (x, y, z),
  - **Cartesian Velocities**: (ẋ, ẏ, ż),
  - **Cartesian Accelerations**: (ẍ, ÿ, z̈),
  - **Cartesian Target positions**: (xₜ, yₜ, zₜ),
  - **Cartesian Target velocities**: (ẋₜ, ẏₜ, żₜ)

**Outputs**:


  - **Interaction Forces**: (Fₓ, Fᵧ, F_z)

![Figure 1: System architecture diagram](Images/transformer_architecture.png)

# Theoretical background

All theoretical explanation of the transformer model, in addition to the selected hyperparameters, sequencing, and batching of data, may be found in the [attached paper](./Paper/Physics-Informed_transformer_ML_Lab.pdf). The paper also describes the data gathering approach with the chirp signal used.

# Managing training

## [Data-driven model training](./main_paper_codes/)

  - [Data-driven model training](./main_paper_codes/InteractionMetaModel_Data_train.py)


Run code with the corresponding datasets for the training, which will save a model 'Interaction_metamodel_physics.pth' or 'Interaction_metamodel_data.pth', for the physics-informed or data-driven models, respectively, every 200 epochs. 

Please note that the models will try to utilize "cuda" if available, if not, the training process may be slow.

# Software requirements

Models were trained and tested in a conda environment utilizing Python 3.12.3 with:

- pandas
- numpy
- pytorch
- wandb
- matplotlib
- scikit-learn
- scipy
- random

Please refer to the official [installation guide](https://www.anaconda.com/docs/tools/working-with-conda/packages/install-packages) to install the mentioned packages in your environment.

# Hardware requirements

While all scripts can run on CPU, execution may be frustratingly slow. For faster training, a GPU is highly recommended. To run the paper's examples, we used a laptop equipped with a Nvidia RTX 4080 GPU.
You can follow the official [CUDA documentation](https://docs.nvidia.com/cuda/index.html) for the installation guide of the cuda-toolkit on your Windows or Linux PC. 

# Robot controller

The work was tested on a Franka Emika Panda, using [ROS2 Humble](https://docs.ros.org/en/humble/index.html) on [Ubuntu 22.04](https://releases.ubuntu.com/jammy/), and in simulation using [MuJoCo3.2.0](https://mujoco.org/). The robot is controlled using a Cartesian impedance controller, expecting as input the desired Cartesian position (xₜ, yₜ, zₜ).

## Simulation Environment and Control

## Real Robot Control



