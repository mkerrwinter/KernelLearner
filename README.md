# K_pipeline_publication

This project consists of a set of files, designed to be run consecutively, that:
- Generate a training set of input-output pairs consisting of GLE solutions and memory kernels respectively.
- Train a network on this training set.
- Make and validate predictions with the trained network.

More details can be found in XXXXXXX INCLUDE ARXIV LINK XXXXXXXX

Below is a description of the whole pipeline, in the order that it should be run.

## 1. generate_data_from_Ks.py
The user must first provide some memory kernels from which to generate a training set. An example of such initial data is given in ./data/example_1D_kernels. $K$ and $\Omega$ are required to define the GLE for $F$, $S$ and $dF0$ are the initial conditions on $F$ and $F'$ respectively, and $t$ is the time grid on which $K$ is defined. The script generate_data_from_Ks.py loads the memory kernels, solves the GLE to produce corresponding F curves, and saves the output in ./data/example_GLE_data.

## 2. generate_F_to_K_datasets.py
The GLE solutions, $F$, and kernels, $K$, from the previous step are loaded. Each $F$ is subjected to multiple realisations of Gaussian noise, producing a large number of noisy curves. PCA is applied to the set of $F$ curves to reduce their dimension to order=15. Pytorch datasets are constructed with inputs $[F_{\text{PCA}},\Omega, F(t_{\text{max}})]$ and output $K$, and saved in .data/F_to_K_data/noise_{noise_str} where {noise_str} is a string defining the strength of noise. The default is '-2'.

## 3. make_param_jsons.py
This script produces a parameter file for a neural network. It can either produce a single parameter file (with hyperparam_search=False, the default value), or multiple files across a range of parameter values. The parameter files are saved in ./models.

## 4. train_network.py
The parameter file from step 3 is loaded and a neural network constructed with the hyperparameters it contains. The network is then trained on the dataset from generate_F_to_K_datasets.py for a number of epochs (default 1000). Every 10 epochs the state of the network is saved.

## 5. extract_kernel_with_net.py
For a set of models sharing a name stub (default name_stub = 'F_to_K_test_model'), this script finds the best network and best epoch (where "best" means the lowest loss on the test set) and uses it to measure a kernel from an unseen $F$ curve. An example unseen $F$ is in ./data/simulation_data'. The corresponding $\Omega$ and time grid are also necessary. The kernel produced by the network is saved in ./extracted_kernels.

## 6. solve_GLE.py
In order to validate the kernel, $K$, measured from $F$ in the previous step, the GLE is solved using $K$ to produce a curve $\hat{F}$. $\hat{F}$ and $F$ are compared and close agreement between the two means the network has measured an accurate kernel.

# Loading a pretrained model
In the publication XXXXXXX INCLUDE ARXIV LINK XXXXXXXXX we describe our work training a model to extract memory kernels from hard sphere MCT data. The trained model is available at XXXXXXXXXXXXXXXXXXXXXXX. The file load_a_state_dict.py demonstrates a minimal example of loading this trained model from the directory ./models/best_trained_model. 


