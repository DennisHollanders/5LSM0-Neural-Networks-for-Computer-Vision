# Final Assignment

This repository contains the work conducted for the 5LSM0 Neural Networks for Computer Vision final assignment. The assignment tries to improve the performance 
of a U-net model on the semantic segmentation of the CityScapes dataset. In doing so the loss functions have been targeted to increase the peak performance while maintaining atleast the same robustness. The main direction of focus for the loss function included tackling the Class Imbalance and the classification around semantic boundaries. 

### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **run_container.sh:** Contains the script for running the container. This File links to the wandb 
  
- **run_main:** Includes the code for building the Docker container. In this file, you only need to change the settings SBATCH (the time your job will run on the server) and ones you need to put your username at the specified location.
  
- **model.py:**  Python file that defines the structure of the neural network model. It contains the classes and functions necessary to construct the architecture for your segmentation task.

- **train.py:** Main Python script for training the model. It contains the data loading, training loop, and validation loop. Customize this script if you need to implement special training procedures or if you want to tweak hyperparameters. 

- **helper.py:** Python script containing custom function, such as the loss function, which are utilized in the train file. 

- **evaluation.py:** File to plot and track the peformance of saved models. 

- **utils.py:** File that changes the existing classes into fewer classes, pushing a certain group of classes to the ignore class

### Authors
- T.J.M. Jaspers
- C.H.B. Claessens
- C.H.J. Kusters
- D.J.C. Hollanders
