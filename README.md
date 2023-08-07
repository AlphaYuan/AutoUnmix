# AutoUnmix
The official Pytorch code for AutoUnmix, an autoencoder-based spectral unmixing method for multi-color fluorescence microscopy imaging

## Run code
Run `main.py` with arguments to train the AutoUnmix on datasets of different fluorophores. 

Run `fine_tune.py.py` to fine-tune AutoUnmix on real biological samples to achieve better reconstruction results.

Run `test.py` to test the performance of the trained model.

Examples are shown in `run.sh`. 

For the experiments of unmixing performance on simulated datasets, we have provided pretrained models in the repository.

`main_simulate.m` is used for generate simulated datasets and is modified from [LUMoS code](https://doi.org/10.1371/journal.pone.0225410).
