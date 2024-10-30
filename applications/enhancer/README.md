# Enhancer Design
For the enhancer experiment, we built on top of the codebase from the [Dirichlet Flow Matching (DirFM) codebase](https://github.com/HannesStark/dirichlet-flow-matching). All relevant files of the enhancer example can be found in `applications/enhancer`.

## Configurations
To use the pretrained models, download the `enhancer.tar.gz` file from zenodo and place
the unpacked `model_weights` folder in this directory or a given specified directory, which
we will refer to as `$parent_dir` (e.g. `parent_dir=discrete_guidance/applications/enhancer/`).
The paths will be defined relative to `$parent_dir`.

You would also need to download the dataset and oracle classifier checkpoint from the DirFM codebase
to compute the evaluation metrics e.g. FBD and target class probability. 
After downloading and extracting the files, place the two directories named `work_dir` and `the_code` under `$parent_dir`.

The file `configs.py` contains the hyperparameters used for training, as well as hyperparameters and path to model checkpoints used for sampling.
Training and sampling setup can be adjusted by modifying this file (see details below).

## Sampling
The script used for sampling with guidance is `scripts/sample_cond.py`. 
By default, the script generates samples for the 8 cell type classes in the manuscript: `[16, 5, 4, 2, 33, 68, 9, 12]`,
each with 5 different temperature values: `[1.0, 0.5, 0.2, 0.1, 0.05]`.
You can adjust these settings by changing `config.target_classes` and `config.guide_temps`
in `configs.py`. The output will be saved to `$parent_dir/outputs/`, which you can adjust by changing `save_dir` in `configs.py`.

The command to reproduce the results for predictor guidance with TAG (PG-TAG) is:
```
    python scripts/sample_cond.py --parent_dir $parent_dir --num_samples 1000 --dt 0.01
``` 
The command to reproduce the results for exact predictor guidance (PG-exact) is:
```
    python scripts/sample_cond.py --parent_dir $parent_dir --num_samples 1000 --dt 0.01 --exact
```
The command to reproduce the results for predictor-free guidance (PFG) is:
```
    python scripts/sample_cond.py --parent_dir $parent_dir --num_samples 1000 --dt 0.01 --pfg --purity
```

## Training
The script used for training a discrete flow model on the enhancer dataset is in `scripts/train_fm.py`.
The commmand to train an unconditional model is:
```
    python scripts/train_fm.py --parent_dir $parent_dir --which_model denoising --fbd
```
The command to train a conditional model (used for predictor-free guidance) is:
```
    python scripts/train_fm.py --parent_dir $parent_dir --which_model denoising --pfg
```
For training the noisy classifier, we found it helpful to train the noisy classifier 
on a distillation dataset obtained by sampling sequences from the unconditional model 
and labeled them with the clean classifier. 
This training set can be created by sampling from a trained unconditional model 
and labeling it with the clean classifier with the following command (which will create 1e6 samples):
```
    python scripts/sample_uncond.py --parent_dir $parent_dir --num_samples 1000000 --dt 0.01 --label
```
The command to train the noisy classifier is (`distill_data_path` is optional. If it is not provided,
then we train the noisy classifier on the same training set of the denoising model):
```
    python scripts/train_fm.py --parent_dir $parent_dir --which_model cls_noisy --distill_data_path $distill_data_path
```
