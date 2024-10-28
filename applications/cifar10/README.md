## Image Modeling

For the image experiment, we built on top of the codebase from https://github.com/andrew-cr/tauLDR/tree/main. 
All references to the files of this experiment can be found in `applications/cifar10`. 

To sample larger batches of images, one can run the script `scripts/sample.py`.
First create a directory `[parent_dir]`.
Then, save the pretrained predictor model checkpoints under `os.path.join([parent_dir], outputs)`.

Below is an example command to perform guided sampling 
(with temperature = 0.3 and with TAG) is:
```
    python applications/cifar10/scripts/sample.py \
        [parent_dir] [pretrained_denoising_model_dirname] [save_sample_dir] \
        --temp 0.3 --guided --approx
```
where [pretrained_denoising_model_dirname] specifies the path to the pretrained 
unconditional denoising model and [save_sample_dir] specifies the path where the
sampled images will be saved. 

For the unconditional denoising model, we used the model checkpoint provided in the repo above.
For the trained noisy classifier, the model checkpoints are provided in `cifar10/` in the zenodo link, unzip and place the `model_weights` directory into `parent_dir`.
