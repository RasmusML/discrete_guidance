## Discrete Guidance

### Overall organization
* The main utility used for training and sampling (both unconditional and with guidance)
    with flow matching is in `src/fm_utils.py`. 
    The file contains code adapted from https://github.com/andrew-cr/discrete_flow_models.
* The code used to produce the results for each experiment are in separate folders in `applications/`
    Running the application specific experiments requires downloading specific
    datasets and model checkpoints. These checkpoints can be found on [our zenodo repository](https://zenodo.org/records/13968379)



### Installation
* First create a new conda environment with, for example, `ENV_NAME=discrete_guidance; conda create -n $ENV_NAME --yes python=3.9`
* Run `./install.sh $ENV_NAME`