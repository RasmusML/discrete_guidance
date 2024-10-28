# Molecule generation application
All relevant files of the molecule generation application can be found in: `applications/molecules/`
All commands below should be ran in the molecule-application directory `/applications/molecules/` and within the project's conda environment denoted by `(conda-env)` in the following.


## Preparation
Download the folders `molecules/data` and `molecules/trained` from the provided location, and place them into `applications/molecules`.


## Molecule-application directory structure
### config_files
The folder `/applications/molecules/config_files` contains the default configurations (e.g., hyperparameters) for model training (`/applications/molecules/config_files/training_defaults.yaml`) and SMILES string (i.e., molecules) generation (`/applications/molecules/config_files/generation_defaults.yaml`).


### data
This folder must be downloaded and placed as described in the section [Preparation](#preparation).

The raw data can be found in `/applications/molecules/data/raw/` with the sub-folder `/applications/molecules/data/raw/qmugs/`.
Within this sub-folder exists a file `/applications/molecules/data/raw/qmugs/summary.txt` that is part of the QMugs dataset ([Isert et al., Sci Data 9, 273 (2022)](https://doi.org/10.1038/s41597-022-01390-7)) and has been downloaded directly from the official QMugs drive: [`https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM`](https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM)

Preprocessing consists of two steps:

* Step (1): First, determine the unique non-stereochemical washed canonical SMILES (nswcs) strings appearing in the raw QMugs dataset (`/applications/molecules/data/raw/qmugs/summary.txt`), and save them as a (one-column) table as `/applications/molecules/data/preprocessed_data/qmugs_unique_nswcs_df.tsv`.  Second, determine molecular properties (#tokens, #rings, #heavy_atoms, logP, molecular-weight) using RDKit for each of the unique nswcs strings and save the
resulting table (nswcs string and corresponding properties per row) as `/applications/molecules/data/preprocessed_data/qmugs_preprocessed_dataset.tsv`.  To re-create these two files run the following script (within the project's conda environment):
```
    [/applications/molecules] (conda-env) python scripts/preprocess.py
```

* Step (2): Filter the nswcs strings of this 'preprocessed dataset' based on their molecular properties and split the resulting set according to the training configurations (for default configurations as used in the article see `/applications/molecules/config_files/training_defaults.yaml`). The resulting (in memory) datasets are used for training and evaluating the models. An explanation for the filter and splitting configurations can be found in the appendix of the article.

Remarks: Step (2) is performed when training the model (as in-memory preprocessing step). If the files described in Step (1) do not exists they will be created and saved (which can be time intensive).


### generated
Generated valid SMILES strings (and their corresponding molecular property values) have been saved in 'samples_table.tsv' found in each folder (corresponding to different generation configurations each) within `/applications/molecules/generated/article/`.

For example, `/applications/molecules/generated/article/logp=-2|n=1000/samples_table.tsv` contains 1000 (=n) valid SMILES strings obtained by (exact) guided generation to the specified lipophilicity value of logP=-2.

### notebooks
The folder `/applications/molecules/notebooks/` contains various Jupyter notebooks used to render article figure panels. These are:
* `Figures_Appendix_DataPreprocessing.ipynb`: Used to render the panels of the figure associated with data preprocessing in the appendix.
* `Figures_Appendix_FrameWorkComparison.ipynb`: Used to render the panels of the figure associated with the comparison of the two frameworks 'Discrete Guidance' (framework presented in the article) and 'DiGress' ([Vignac et al., ICLR 2023](https://arxiv.org/abs/2209.14734)) that can be found in the appendix. 
* `Figures_Appendix_WideRangeGeneration.ipynb`: Used to render the panels of the two figures containing property-histograms across a wide range of specified property values.
* `Figures_MainText.ipynb`: Used to render the panels of the figure associated with molecules generation in the main text of the article.

### scripts
Scripts for preprocessing, training, and generation can be found in `/applications/molecules/scripts/`. These are:
* `generate.py`: Used for generation as described in the section [Generation](#generation).
* `preprocess.py`: Used for preprocessing as described in the section [data](#data).
* `train.py`: Used for training as described in the section [Training](#training).


### src
Source files used for this application can be found in `/applications/molecules/src/`.

### trained
This folder must be downloaded and placed as described in the section [Preparation](#preparation).
Model weights and archived files associates with training the models (e.g., log-files) are located in the folders (where each correspond to training with different configurations) within `/applications/molecules/trained/`:

* `CTFM/`: Folder corresponding to the trained 'continuous-time flow model' and its (continuous-time) prediction models. 

* `DTDM/`: Folder corresponding to the trained 'discrete-time diffusion model' and its (discrete-time) prediction models. 

The model weights for the 'denoising', 'logp_predictor', 'num_rings_predictor', and 'num_heavy_atoms_predictor' (not used in article) model can be found in the sub-folders `/applications/molecules/trained/CTFM/models_saved/` and `/applications/molecules/trained/DTDM/models_saved/`.


## Instructions for training and generation
### Training
The script used for training a continuous-time discrete-space flow model and property-predictor models is located in `/applications/molecules/scripts/train.py`.

The default training configuration file can be found in `/applications/molecules/config_files/training_defaults.yaml`.

To train all (continuous-time) models for the default training configurations, run the following command:
```
    [/applications/molecules] (conda-env) python scripts/train.py -c config_files/training_defaults.yaml -m "all"
```
This command will reproduce the (continuos-time) models used for downstream generation of molecules presented in our article for our method (Discrete Guidance) that can be found in the folder `/applications/molecules/trained/CTFM/`.

To train only the (continuous-time) denoising model for the default training, run the following command:
```
    [/applications/molecules] (conda-env) python scripts/train.py -c config_files/training_defaults.yaml -m "denoising_model"
```
To train all (continuous-time) models while overriding some default training configurations (here training the denoising model for 10 epochs), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/train.py -c config_files/training_defaults.yaml -m "all" -o "training.denoising_model.num_epochs=10"
```
To train all (continuous-time) models while overriding some default training configurations (here training the denoising model for 10 epochs AND only including molecules with less then 4 rings in the data), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/train.py -c config_files/training_defaults.yaml -m "all" -o "training.denoising_model.num_epochs=10|data.preprocessing.filter_range_dict.num_rings=[0,3]"
```
All of these will create an individual run folder in the directory `/applications/molecules/trained` in the format `YYYY-MM-DD/<overrides>`, while using `<overrides>=no_overrides` if no default configurations were overriden.
Note that the run folder name does not include the model specification and thus will be overwritten when training with the same configurations but different model specification.

Remark: We have only described training of the continuous-time model(s) above. To train the discrete-time model(s) one can run the same commands as above, while stacking the overrides as "<overrides>|num_timesteps=1000".

For example, to train all (discrete-time discrete-space) models (with 1000 discrete time steps) for the default training configurations (except that there are 1000 discrete time steps), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/train.py -c config_files/training_defaults.yaml -m "all" -o "num_timesteps=1000"
```
This command will reproduce the (discrete-time) models used for downstream generation of molecules mentioned in our article (as baselines) that can be found in the folder `/applications/molecules/trained/DTDM/`.


### Generation
The script used for training a discrete flow model and property-predictor models is located in:
`/applications/molecules/scripts/generate.py`

The default generation configuration file is located in `/applications/molecules/config_files/generation_defaults.yaml`.

To unconditionally generate 1000 valid molecules with the default generation configurations (and the continuos-time discrete-space flow model), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/generate.py -c config_files/generation_defaults.yaml -n 1000
```

To generate 1000 valid molecules guided to one ring ("num_rings=1") with the default generation configurations (using Discrete Guidance with the continuous-time models), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/generate.py -c config_files/generation_defaults.yaml -n 1000 -p "num_rings=1"
```
To generate 1000 valid molecules guided to a lipophilicity value of 10 ("logp=10") with the default generation configurations (using Discrete Guidance with the continuous-time models), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/generate.py -c config_files/generation_defaults.yaml -n 1000 -p "logp=10"
```
To unconditionally generate 1000 valid molecules but record the the (RDKit-determined) ground truth and predicted lipophilicity values with the default generation configurations (using Discrete Guidance with the continuous-time models), pass "logp=None" for the property and thus run the following command:
```
    [/applications/molecules] (conda-env) python scripts/generate.py -c config_files/generation_defaults.yaml -n 1000 -p "logp=None"
```
To generate 1000 valid molecules guided to one ring ("num_rings=1") (using Discrete Guidance with the continuous-time models) while overriding some of the default generation configurations (e.g., using Taylor-approximated gradients and a batch size of 250), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/generate.py -c config_files/generation_defaults.yaml -n 1000 -p "num_rings=1" -o "sampler.grad_approx=True|sampler.batch_size=250"
```
All of these will create a run folder in the directory `/applications/molecules/generated` in the format `YYYY-MM-DD/<overrides>`, while using `<overrides>=no_overrides` if no default configurations were overriden.

Remark: We have only described generation using the continuous-time model(s) above. To generate using the discrete-time model(s) one can run the same commands as above, while stacking the overrides as "<overrides>|sampler.grad_approx=True|trained_run_folder_dir=./trained/DTDM".

For example, to unconditionally generate 1000 valid molecules with the default generation configurations and the discrete-time discrete-space diffusion model (that has to be guided using the gradient approximation), run the following command:
```
    [/applications/molecules] (conda-env) python scripts/generate.py -c config_files/generation_defaults.yaml -n 1000 -o "sampler.grad_approx=True|trained_run_folder_dir=./trained/DTDM"
```