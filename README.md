# Varational meta-Learning reduced-order neural model structures

This repository contains the Python code to reproduce the results of the paper *Variational meta-learning inference for low dimensional neural system identification*

In this repository we extend the model-free in-context learning architecture introduced here [Manifold meta-learning for reduced-complexity neural system identification](https://arxiv.org/abs/2504.11811) which introduced a model-free meta-learning architecture for system identification in which a meta-model learns to represent an entire class of dynamical systems through a compact latent manifold.
We focus on one innovations: formulating the learning task within a variational framework.



## Main files

The main files are divided in two folders one for the static and the other for the dynamic one:
 
### Static example
* [deterministic_hypernet.ipynb](meta_train.ipynb): Learn the reduced-order architecture for regressing sines functions
* [vae_hypernet.ipynb](meta_train.ipynb): Learn the probabilistic reduced-order for regressing sines functions


### Dynamic example
* [meta_train.ipynb](meta_train.ipynb): Learn the reduced-order architecture on the meta dataset
* [meta_train_VAE.ipynb](meta_train.ipynb): Learn the probabilistic reduced-order architecture on the meta dataset
* [train_reduced_mc.ipynb](train_reduced_mc.ipynb): Train reduced-order models on sequences of different lengths
* [train_reduced_mc_VAE.ipynb](train_reduced_mc.ipynb): Train the probabilistic reduced-order models on sequences of different lengths
* [analyze_mc.ipynb](analyze_mc.ipynb): Analyze Monte Carlo experiments and obtain paper figures

It is preferrable to run the training files from the command line through ipython:

``
$ ipython 
``

``
run <file_name>
``

Additional python files:

* [ae.py](ae.py) Autoencoder model blocks
* [neuralss.py](neuralss.py) Neural state-space base architecture
* [lr.py](lr.py) Learning rate scheduling
* [plot_utils.py](plot_utils.py) Matplotlib settings for good-looking plots in papers

# Hardware requirements
While all the scripts can run on CPU, execution may be frustratingly slow. For faster training, a GPU is highly recommended.
To run the paper's examples, we used a server equipped with an nVidia RTX 3090 GPU.


# Citing

If you find this project useful, we encourage you to:

* Star this repository :star: 

