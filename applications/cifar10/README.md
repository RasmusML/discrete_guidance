# Image Modeling

For the image experiment, we built on top of the codebase from the [tauLDR codebase](https://github.com/andrew-cr/tauLDR/tree/main). 
All references to the files of this experiment can be found in `applications/cifar10`. 

The model checkpoints for both the pretrained denoising model from the tauLDR codebase and a noisy classifier we trained are provided in the `cifar10.tar.gz` file in zenodo. Download this file, then place the unpacked `model_weights` folder in this directory or a given directory you specified, which we will refer to as `$parent_dir` (e.g. `parent_dir=discrete_guidance/applications/enhancer/`). The paths for this application will be defined relative to `$parent_dir`.

The script used for sampling with guidance is `scripts/sample.py`.
Below is an example command to draw 100 samples for each of the 10 classes with temperature = 0.3 and with PG-TAG:
```
    python scripts/sample.py $parent_dir $save_sample_dir --temp 0.3 --guided --approx --num_samples 100
```
where `$save_sample_dir` specifies the path where the
sampled images will be saved. 

We also provided some utilities for training a continuous-time discrete diffusion model based on the tauLDR codebase. These utilities are not used in our manuscript and are not thoroughly test. They are for reference purpose only.