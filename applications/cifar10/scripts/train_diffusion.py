"""
Script to train denoising diffusion model 
or a noisy property model under the diffusion process
"""
import torch
import argparse

from applications import bookkeeping
from applications.cifar10 import (
    managers,
    forward_processes,
    training,
    utils
)
from applications.cifar10.configs import (
    get_cifar10_train_config, 
    get_cifar10_noisy_classifier_train_config,
    get_cifar10_test_config
)
from applications.cifar10.dataset import DiscreteCIFAR10
from applications.cifar10.models import ImageX0PredBase, count_params
from applications.cifar10.networks import UNetClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', type=str, help='Path to the parent directory where model checkpoints and outputs are saved')
    parser.add_argument('which_model', type=str, help='which model to train, either denoising or property')
    args = parser.parse_args()
    parent_dir = args.parent_dir
    which_model = args.which_model
    
    # Load the relevant training config files
    if which_model == 'denoising':
        cfg = get_cifar10_train_config(parent_dir)
    elif which_model == 'property':
        cfg = get_cifar10_noisy_classifier_train_config(parent_dir)
    else:
        raise ValueError("Training model should be 'denoising' or 'property'")

    # Test config is only relevant for specifying the dataset
    # so we use the same one regardless of the model
    test_cfg = get_cifar10_test_config(parent_dir)

    # Set up logging
    save_dir, checkpoint_dir, config_dir = \
        bookkeeping.create_experiment_folder(
            cfg.save_location,
            cfg.experiment_name,
    )
    bookkeeping.save_config_as_yaml(cfg, config_dir)
    writer = bookkeeping.setup_tensorboard(save_dir, 0)

    # Define datasets
    train_dataset = DiscreteCIFAR10(cfg)
    test_dataset = DiscreteCIFAR10(test_cfg)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)

    # Define models
    forward_process = forward_processes.GaussianTargetRateForwardProcess(cfg)
    if which_model == 'denoising':
        denoising_model = ImageX0PredBase(cfg)
        denoising_optimizer = torch.optim.Adam(
            denoising_model.parameters(), lr=cfg.optimizer.lr)
        property_model = None
        property_optimizer = None
        print(f"Denoising model number of parameters: {count_params(denoising_model):.3E}")
    else:
        property_model = UNetClassifier(cfg)
        property_optimizer = torch.optim.Adam(
            property_model.parameters(), lr=cfg.optimizer.lr)
        denoising_model = None
        denoising_optimizer = None
        print(f"Property model number of parameters: {count_params(property_model):.3E}")
    
    # Initialize diffusion manager
    diffusion_manager = managers.DiscreteDiffusionManager(
        cfg, forward_process, 
        denoising_model, denoising_optimizer,
        property_model, property_optimizer, 
        time_encoder=None, debug=False)

    # Train the model
    utils.set_random_seed(43)
    training.train_manager_on_cifar(
        diffusion_manager, train_dataloader, 
        writer=writer, cfg=cfg,
        which_model=which_model,
        valid_dataloader=test_dataloader,
        checkpoint_dir=checkpoint_dir 
    )

if __name__ == '__main__':
    main()