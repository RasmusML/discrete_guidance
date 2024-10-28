import torch
from pathlib import Path
import numpy as np
import os
import time
import argparse
from PIL import Image

import applications.bookkeeping as bookkeeping
from applications.cifar10 import (
    managers,
    forward_processes,
    model_utils,
    utils,
    models
)
from applications.cifar10.configs import get_cifar10_test_config
from applications.cifar10.networks import UNetClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', type=str, help='Path to the parent directory where model checkpoints and outputs are saved')
    parser.add_argument('pretrained_denoising_model_dirname', type=str, help='Name of the directory where the pretrained denoising model is stored')
    parser.add_argument('save_samples_dir', type=str, help='The directory to save the generated samples')
    parser.add_argument('sampler_name', type=str, nargs='?', default='tau_leaping', help='Which sampler to use')
    parser.add_argument('num_samples', type=int, nargs='?', default=5000, help='Total number of samples, for guided this is the number of samples per class')
    parser.add_argument('-t', '--temp', type=float, default=0.1, help='Set the temperature value.')
    parser.add_argument('-g', '--guided', help='If true, do classifier guidance', action='store_true')
    parser.add_argument('-a', '--approx', help='If true, do classifier guidance with gradient approximation', action='store_true')
    args = parser.parse_args()
    parent_dir = args.parent_dir
    pretrained_denoising_model_dirname = args.pretrained_denoising_model_dirname
    save_samples_dir = args.save_samples_dir
    temp = args.temp
    guided = args.guided
    grad_approx = args.approx
    sampler_name = args.sampler_name
    total_num_samples = args.num_samples

    if not os.path.exists(save_samples_dir):
        os.makedirs(save_samples_dir)
        print(f"Directory '{save_samples_dir}' created")
    else:
        print(f"Directory '{save_samples_dir}' already exists and will be overwritten")

    # Get config file for sampling
    eval_cfg = get_cifar10_test_config(
        parent_dir, pretrained_denoising_model_dirname, save_samples_dir, sampler_name
    )
    device = torch.device(eval_cfg.device)

    # Load trained denoising model
    denoising_model_train_cfg = bookkeeping.load_ml_collections(
        Path(eval_cfg.denoising_model_train_config_path))
    for item in eval_cfg.train_config_overrides:
        bookkeeping.set_in_nested_dict(denoising_model_train_cfg, item[0], item[1])
    denoising_model = model_utils.create_model(denoising_model_train_cfg, device)
    loaded_state = torch.load(
        Path(eval_cfg.denoising_model_checkpoint_path),
        map_location=device)
    modified_model_state = bookkeeping.remove_module_from_keys(loaded_state['model'])
    denoising_model.load_state_dict(modified_model_state)
    denoising_model.eval()

    # Define the forward process
    forward_process = forward_processes.GaussianTargetRateForwardProcess(
        denoising_model_train_cfg
    )

    # Load trained property model
    print(f'Generate {total_num_samples} samples (if guided, this is per class)')
    print(f'Number of steps: {eval_cfg.sampler.num_steps}, sampler: {eval_cfg.sampler.name}')
    print(f'Loaded denoising model from: {eval_cfg.denoising_model_checkpoint_path}')
    if guided:
        print(f'Guided diffusion with temp={temp}')
        property_model_train_cfg = bookkeeping.load_ml_collections(
            Path(eval_cfg.property_model_train_config_path))
        property_model = UNetClassifier(property_model_train_cfg)
        loaded_state = torch.load(
            Path(eval_cfg.property_model_checkpoint_path),
            map_location=device)
        property_model.load_state_dict(loaded_state['model'])
        property_model.eval()
        print(f'Loaded property model from: {eval_cfg.property_model_checkpoint_path}')
    else:
        property_model = None

    # Define diffusion manager for sampling
    diffusion_manager = managers.DiscreteDiffusionManager(
        eval_cfg, forward_process, 
        denoising_model=denoising_model,
        property_model=property_model
    )

    # Generate samples by batch
    batch_size = 100
    counter = 0
    while True:
        start_time = time.time()

        if not guided:
            sampler_outputs = diffusion_manager.generate(
                num_samples=batch_size,
                num_intermediate_samples=0,
                guide_temp=temp
            )
            samples = sampler_outputs['x_0']
            samples = samples.reshape(batch_size, 3, 32, 32)
            samples_uint8 = samples.astype(np.uint8)
            for i in range(samples.shape[0]):
                path_to_save = os.path.join(eval_cfg.save_samples_dir, f'{i + counter}.png')
                img = Image.fromarray(utils.imgtrans(samples_uint8[i]))
                img.save(path_to_save)
        
        else:
            if grad_approx:
                # For gradient approximation, sample conditioning on all classes
                all_classes = np.arange(10)
                for y in all_classes:
                    batch_y = torch.full((batch_size,), y).to(device)
                    sampler_outputs = diffusion_manager.generate(
                        num_samples=batch_size,
                        num_intermediate_samples=0,
                        batch_y=batch_y,
                        classifier_guidance=True,
                        grad_approx=True,
                        guide_temp=temp
                    )
                    samples = sampler_outputs['x_0']
                    samples = samples.reshape(batch_size, 3, 32, 32)
                    samples_uint8 = samples.astype(np.uint8)
                    for i in range(samples.shape[0]):
                        path_to_save = os.path.join(eval_cfg.save_samples_dir, f'{y}-{i + counter}.png')
                        img = Image.fromarray(utils.imgtrans(samples_uint8[i]))
                        img.save(path_to_save)
            else:
                # For exact sampling, just do one class
                y = 1
                batch_y = torch.full((batch_size,), y).to(device)
                sampler_outputs = diffusion_manager.generate(
                    num_samples=batch_size,
                    num_intermediate_samples=0,
                    batch_y=batch_y,
                    classifier_guidance=True,
                    grad_approx=False
                )
                samples = sampler_outputs['x_0']
                samples = samples.reshape(batch_size, 3, 32, 32)
                samples_uint8 = samples.astype(np.uint8)
                for i in range(samples.shape[0]):
                    path_to_save = os.path.join(eval_cfg.save_samples_dir, f'{y}-{i + counter}.png')
                    img = Image.fromarray(utils.imgtrans(samples_uint8[i]))
                    img.save(path_to_save)

        counter += batch_size
        print(f"{counter} out of {total_num_samples} generated, time: {time.time() - start_time:.2f} seconds")
        if counter >= total_num_samples:
            break


if __name__ == '__main__':
    main()