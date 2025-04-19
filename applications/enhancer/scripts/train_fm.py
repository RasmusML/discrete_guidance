"""
Script to train denoising diffusion model 
or a noisy property model under the diffusion process
"""

import numpy as np
import torch
import argparse
import os

import applications.bookkeeping as bookkeeping
import applications.enhancer.utils as utils
from applications.enhancer.configs import get_enhancer_config
from applications.enhancer.dataset import EnhancerDataset, GeneratedDataset
from applications.enhancer.models import CNNModel, count_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="data/enhancer",
        help="Path to the parent directory where model checkpoints and outputs are saved",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="denoising",
        help="which model to train, either denoising, cls_noisy or cls_clean",
    )
    parser.add_argument(
        "--pfg",
        help="If true, train with classifier-free guidance",
        action="store_false",
    )
    parser.add_argument(
        "--fbd", help="If true, compute fbd at train time", action="store_true"
    )
    parser.add_argument(
        "--distill_data_path",
        type=str,
        nargs="?",
        default=None,
        help="If provided, train noisy classifier on a distillation dataset",
    )
    parser.add_argument(
        "--discrete_time",
        help="If true, train discrete time models",
        action="store_true",
    )
    args = parser.parse_args()
    parent_dir = args.parent_dir
    which_model = args.which_model
    discrete_time = args.discrete_time
    assert which_model in ["denoising", "cls_noisy", "cls_clean"]

    cls_free_guidance = args.pfg
    train_fbd = args.fbd
    distill_data_path = args.distill_data_path

    if which_model != "denoising":
        train_fbd = False
    if which_model != "cls_noisy":
        distill_data_path = None

    print(
        f"Train {which_model} model, PFG={cls_free_guidance}, FBD early stopping={train_fbd}"
    )

    # Configs for training
    cfg = get_enhancer_config(
        parent_dir=parent_dir,
        state="train",
        which_model=which_model,
        cls_free_guidance=cls_free_guidance,
        train_fbd=train_fbd,
        distill_data_path=distill_data_path,
        discrete_time=discrete_time,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Configs for evaluation
    eval_cfg = get_enhancer_config(
        parent_dir=parent_dir, state="eval", which_model=which_model
    )
    device = cfg.device

    # Set up logging directory
    save_dir, checkpoint_dir, config_dir = bookkeeping.create_experiment_folder(
        cfg.save_location,
        cfg.experiment_name,
    )
    bookkeeping.save_config_as_yaml(cfg, config_dir)
    writer = bookkeeping.setup_tensorboard(save_dir, 0)

    # Define datasets
    if distill_data_path is not None:
        # Use synthetic dataset for training a noisy classifier
        # to distill a clean classifier
        synthetic_dataset = np.load(cfg.distill_data_path)
        seqs, labels = synthetic_dataset["x"], synthetic_dataset["y"]
        dataset = GeneratedDataset(x=seqs, y=labels, device=device)
        # Use 80:20 train val split
        # Use fixed random seed and save validation indices
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(cfg.seed)
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        val_indices = np.array(valid_dataset.indices)
        np.save(os.path.join(save_dir, "val_indices.npy"), val_indices)
        print(
            f"Train noisy classifier on distillation dataset from {distill_data_path}: \
            train size: {len(train_dataset)}, val size: {len(valid_dataset)}"
        )
    else:
        # Use training and validation datasets
        train_dataset = EnhancerDataset(cfg, split="train", n_seqs=20)
        valid_dataset = EnhancerDataset(eval_cfg, split="valid", n_seqs=20)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0
    )

    # Define models
    model = CNNModel(cfg).to(device)
    print(f"{which_model} model #parameters: {count_params(model):.3E}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    if train_fbd:
        # If computing FBD at train time, also need clean classifier
        # Load clean classifier
        cls_clean_model_cfg = get_enhancer_config(
            parent_dir=parent_dir, state="eval", which_model="cls_clean"
        )
        cls_clean_model = CNNModel(cls_clean_model_cfg).to(device)
        # The ckpt from DFM has an extra 'model.' in each key
        # We rename the state dict so we can load into CNNModel
        cls_ckpt = torch.load(eval_cfg.cls_clean_model_checkpoint_path)
        state_dict = dict()
        for k, v in cls_ckpt["state_dict"].items():
            new_k = k[6:]
            state_dict[new_k] = v
        cls_clean_model.load_state_dict(state_dict)
    else:
        cls_clean_model = None

    # Train the model
    utils.set_random_seed(cfg.seed)
    utils.train_model_on_dataset(
        cfg,
        train_dataloader,
        model,
        optimizer,
        writer=writer,
        which_model=which_model,
        valid_dataloader=valid_dataloader,
        checkpoint_dir=checkpoint_dir,
        cls_clean_model=cls_clean_model,
        eval_cfg=eval_cfg,
    )


if __name__ == "__main__":
    main()
