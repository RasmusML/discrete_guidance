"""
Generate unconditional samples
Optionally label the samples with a clean classifer
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import os
import json

import src.fm_utils as fm_utils
import applications.bookkeeping as bookkeeping
import applications.enhancer.utils as utils
from applications.enhancer.configs import get_enhancer_config
from applications.enhancer.models import CNNModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="data/enhancer/",
        help="Path to the parent directory where model checkpoints and outputs are saved",
    )
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="euler",
        help="Which sampler to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to draw for each class",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time interval for Euler sampling",
    )
    parser.add_argument(
        "--label",
        help="If true, label the generated samples with a clean classifier",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    parent_dir = args.parent_dir
    sampler_name = args.sampler_name
    num_samples = args.num_samples
    dt = args.dt
    label_samples = args.label

    # Load the config file for evaluation
    # Some arguments are provided in the command line for convenience
    # To change the other arguments, need to modify the config file
    eval_cfg = get_enhancer_config(
        parent_dir=parent_dir,
        state="eval",
        which_model="denoising",
        sampler_name=sampler_name,
        dt=dt,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(eval_cfg.device)
    print(f"Sample from denoising model at {eval_cfg.denoising_model_checkpoint_path}")
    # Set up logging directory
    save_dir, _, _ = bookkeeping.create_experiment_folder(
        eval_cfg.save_location,
        eval_cfg.experiment_name,
    )
    bookkeeping.save_config_as_yaml(eval_cfg, save_dir)
    writer = bookkeeping.setup_tensorboard(save_dir, 0)

    # Load trained denoising model
    denoising_model_train_cfg = bookkeeping.load_ml_collections(
        Path(eval_cfg.denoising_model_train_config_path)
    )
    denoising_model = CNNModel(denoising_model_train_cfg).to(device)
    denoising_ckpt_path = eval_cfg.denoising_model_checkpoint_path
    loaded_state = torch.load(denoising_ckpt_path, map_location=device)
    denoising_model.load_state_dict(loaded_state["model"])

    # Load clean classifier for evaluation
    # and optionally label the generated samples for distillation
    use_classifier = False

    if use_classifier:
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

    # Parse sampling configs
    S = eval_cfg.data.S
    D = eval_cfg.data.shape
    stochasticity = eval_cfg.sampler.noise
    x1_temp = eval_cfg.sampler.x1_temp
    purity_temp = eval_cfg.sampler.purity_temp
    do_purity_sampling = eval_cfg.sampler.do_purity_sampling
    argmax_final = eval_cfg.sampler.argmax_final
    max_t = eval_cfg.sampler.max_t
    batch_size = eval_cfg.sampler.batch_size
    mask_idx = S - 1

    # Log configs
    config_str = json.dumps(eval_cfg.to_dict(), indent=2)
    config_str = config_str.replace("\n", "\n\n")
    writer.add_text("Config", config_str)

    # Generate unconditional samples
    denoising_model_func = lambda xt, t: denoising_model((xt, None), t)
    samples = fm_utils.flow_matching_sampling(
        num_samples,
        denoising_model_func,
        S=S,
        D=D,
        device=device,
        mask_idx=mask_idx,
        dt=dt,
        batch_size=batch_size,
        stochasticity=stochasticity,
        argmax_final=argmax_final,
        max_t=max_t,
        x1_temp=x1_temp,
        do_purity_sampling=do_purity_sampling,
        purity_temp=purity_temp,
        argmax_sample=True,
    )

    if use_classifier:
        metrics = utils.print_eval_metrics(samples, cls_clean_model, eval_cfg)
        writer.add_text("Uncond FBD", f"FBD(uncond): {metrics['fbd_uncond']}")
    
    if label_samples:
        labels = utils.get_labels_from_cls_labeler_model(samples, cls_clean_model)
        with open(os.path.join(save_dir, f"samples_uncond-labeled.npz"), "wb") as f:
            np.savez(f, x=samples, y=labels)
    else:
        with open(os.path.join(save_dir, f"samples_uncond.npy"), "wb") as f:
            np.save(f, samples)


if __name__ == "__main__":
    main()
