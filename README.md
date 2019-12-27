# Protein structure prediction with GANs

A repository with implementations of GANs:

1. Vanilla GAN
2. DCGAN
3. DCGANs for protein structure prediction based on the paper "Generative Modeling for Protein Structures" by Anand and Huang (2018).
4. Modified DCGANs improving the lacking quality of the original DCGAN architectures by Anand and Huang.
5. A residual network (ResNet) for residue fragment length of 256.

# Getting started
 
## Installing

Install the packages/libraries that are enlisted in the `requirements.txt` file. You can also install them with the commands listed below:

```
pip install -r requirements.txt
```
Or if chosing an environment in Anaconda:
```
conda install --file requirements.txt
```

## Data sets

In order to run the model `DCGAN`, download the CELEBA and place it in `data/celeba`. For the proteins, download the CASP datasets as instructed in the [ProteinNet](https://github.com/aqlaboratory/proteinnet) project. These should then be placed in `data/proteins`.

# Running the repository

## Training a model

The parameters for the models are uniquely set in the file `model_params.py`. Please see in the file the allowed data sets of each model (or the jupyter notebooks).

Train models either in the jupyter notebooks or the command line in a terminal. For command line, the arguments are below:

```
python solver.py --model <model name> --dataset <data set> [--save_files] [--save_model_state] [--max_sequence_length] <positive int> [--residue_fragments] <64/128/256>
```


For more help for training, retieve the information about arguments with:
```
python solver.py --help
```

## Producing the plots

After having trained a model, the model can be loaded in the corresponding jupyter notebook. The plots can then be produced in the notebook. **Plots cannot be produced from the script**.
