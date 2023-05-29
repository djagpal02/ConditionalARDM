## Conditional ARDM - PyTorch Implementation

This repository presents a conditional version of the ARDM model https://arxiv.org/abs/2110.02037, with official code available at: https://github.com/google-research/google-research/tree/master/autoregressive_diffusion. This version empowers the ARDM to model P(X|Z) instead of the typical P(X). For this to work, Z must be of the same shape as X. In case Z is not of the same shape, it can be projected to the same shape to ensure the model's functionality, i.e., P(X|projection(Z)).

The modification allows for additional conditioning, expanding the use cases for ARDM into the conditional modelling space. Specifically, we concatenate X, Z, and the mask before the initial projection, a shift from the original model where only X and the mask are concatenated.

This conditional model enables us to sample high-quality images based on blurry versions of the images we intend to generate. This results in the model conditioning on the blurry input as desired.

To provide versatility, the code allows you to turn off the conditioning and use the model as a standard density model, as intended by the original authors.

Our code is implemented in PyTorch, unlike the original code which is in JAX.

We provide a comprehensive pipeline for training, testing, and sampling that is efficiently parallelized. Training can be executed across multiple GPUs using PyTorch's Distributed Data Parallel (DDP), and testing is streamlined by dispatching batches of timesteps to each GPU for independent processing, with results combined at the end. Although sampling cannot be parallelized directly, its speed can be enhanced by approximation via sampling more dimensions per forward pass (full sampling just one dimension per forward pass).

### Features

- **Conditional ARDM model:** Enables ARDM to model P(X|Z), expanding its use cases into conditional modeling space.
- **Versatile Conditioning:** Allows for conditioning to be turned off, returning the model to a standard density model.
- **PyTorch Implementation:** Unlike the original JAX code, our implementation uses PyTorch.
- **Parallelized Pipelines:** We provide highly parallelized pipelines for training and testing.

This repository is designed to help you generate higher-quality images based on your conditional inputs and model condtional densities for complex datasets.
