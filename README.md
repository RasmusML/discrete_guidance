## Discrete Guidance
Code release for "Unlocking Guidance for Discrete State-Space Diffusion and Flow Models" ([arXiv](https://arxiv.org/abs/2406.01572))

### Overall organization
* The main utility used for training and sampling (both unconditional and with guidance)
    with flow matching is in `src/fm_utils.py`. 
    The file contains code adapted from https://github.com/andrew-cr/discrete_flow_models.
* The code used to produce the results for each experiment are in separate folders in `applications/`
    Running the application specific experiments requires downloading specific
    datasets and model checkpoints. These checkpoints can be found on [our zenodo repository](https://zenodo.org/records/13968379)
    Each application folder contains its own readme file that describe the setup in more details.

### Installation
* First create a new conda environment with, for example, `ENV_NAME=discrete_guidance; conda create -n $ENV_NAME --yes python=3.9`
* Run `conda activate $ENV_NAME; ./install.sh $ENV_NAME`

### Key Hyperparameter Considerations
* The number of timesteps is controlled by the parameter, `dt`. We recommend using a `dt` such that `1/dt` is at least as big as the number of dimensions as your system (e.g. `dt <= 0.01` for a protein of length 100). In general, smaller values of `dt` should result in better performance at the cost of taking a longer time to run. 

* The stochasticity parameter used in flow matching can play a large role in sample quality. We recommend trying out different settings of this parameter such as 0, 1, 10, and 100. 

* Ultimately, the quality of the noisy classifiers substantially impacts the ability of guidance to achieve the desired goal. Care should be taken to ensure that the noisy classifiers are well-trained.
