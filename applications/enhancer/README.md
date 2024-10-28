# Enhancer Design
For the enhancer experiment, we built on top of the codebase from the [Dirichlet Flow Matching (DirFM) codebase](https://github.com/HannesStark/dirichlet-flow-matching). All relevant files of the enhancer example can be found in `applications/enhancer`.
All commands below should be ran in the home directory `gdd/`.

## Sampling
The script used for sampling with guidance is `scripts/sample_cond.py`. 
To sample from the pretrained model, first create a directory `[parent_dir]`. Then, save the pretrained model checkpoints under `os.path.join([parent_dir], "outputs")`.
The command to reproduce the results for predictor guidance with TAG (PG-TAG) is:
```
    python applications/enhancer/scripts/sample_cond.py --parent_dir [parent_dir] --num_samples 1000 --dt 0.01
``` 
The command to reproduce the results for exact predictor guidance (PG-exact) is:
```
    python applications/enhancer/scripts/sample_cond.py --parent_dir [parent_dir] --num_samples 1000 --dt 0.01 --exact
```
The command to reproduce the results for predictor-free guidance (PFG) is:
```
    python applications/enhancer/scripts/sample_cond.py --parent_dir [parent_dir] --num_samples 1000 --dt 0.01 --pfg --purity
```

## Configurations
`configs.py` contains the hyperparameters used for training, as well as hyperparameters and path to model checkpoints used for sampling.
The model checkpoints are provided in `enhancer/` in the zenodo link, unzip it into `parent_dir`.

Note that you would also need to download the data from the DirFM codebase for training and the oracle classifier checkpoint for evaluation. Both should be placed under `parent_dir`. 


## Training
The script used for training a discrete flow model on the enhancer dataset is in `scripts/train_fm.py`.
The commmand to train an unconditional model is:
```
    python applications/enhancer/scripts/train_fm.py --parent_dir [parent_dir] --which_model denoising --fbd
```
The command to train a conditional model (used for predictor-free guidance) is:
```
    python applications/enhancer/scripts/train_fm.py --parent_dir [parent_dir] --which_model denoising --pfg
```
For training the noisy classifier, we found it helpful to train the noisy classifier 
on a distillation dataset obtained by sampling sequences from the unconditional model 
and labeled them with the clean classifier.
The distillation dataset we used is `data/samples_uncond-labeled-oracle.npz` under
`enhancer/` in the zenodo link. 
The command to train the noisy classifier is (`distill_data_path` is optional. If it is not provided,
then we train the noisy classifier on the same training set of the denoising model):
```
    python applications/enhancer/scripts/train_fm.py --parent_dir [parent_dir] --which_model cls_noisy --distill_data_path [distill_data_path]
```
You can also create your own training set by sampling from a trained unconditional model 
and labeling it with the clean classifier with the following command (which will create 10000 samples):
```
    python applications/enhancer/scripts/sample_uncond.py --parent_dir [parent_dir] --num_samples 10000 --dt 0.01 --label
```