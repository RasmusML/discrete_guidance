"""
Generate conditional samples
Either with predictor guidance or predictor-free guidance
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import os
import pickle
import json
import time

import src.fm_utils as fm_utils
import src.digress_utils as digress_utils

import applications.bookkeeping as bookkeeping
import applications.enhancer.utils as utils
from applications.enhancer.configs import get_enhancer_config
from applications.enhancer.models import CNNModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="/data/gdd/enhancer/",
        help="Path to the parent directory where model checkpoints and outputs are saved",
    )
    parser.add_argument(
        "--sampler_name", type=str, default="euler", help="Which sampler to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to draw for each class",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Time interval for Euler sampling"
    )
    parser.add_argument(
        "--num_replicates",
        type=int,
        default=1,
        help="Number of independent sampling replicates",
    )
    parser.add_argument(
        "--pfg",
        help="If true, sample with predictor free guidance",
        action="store_true",
    )
    parser.add_argument(
        "--purity", help="If true, do purity sampling", action="store_true"
    )
    parser.add_argument(
        "--exact",
        help="If true, do exact predictor guidance instead of tag",
        action="store_true",
    )
    parser.add_argument(
        "--discrete_time",
        help="If true, train discrete time models",
        action="store_true",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=100,
        help="Number of timesteps for discrete time models",
    )

    args = parser.parse_args()
    parent_dir = args.parent_dir
    sampler_name = args.sampler_name
    num_samples = args.num_samples
    dt = args.dt
    cls_free_guidance = args.pfg
    do_purity_sampling = args.purity
    num_replicates = args.num_replicates
    use_tag = not args.exact
    discrete_time = args.discrete_time
    num_timesteps = args.num_timesteps

    # Load the config file for evaluation
    # Some arguments are provided in the command line for convenience
    # To change the other arguments, need to modify the config file
    eval_cfg = get_enhancer_config(
        parent_dir=parent_dir,
        state="eval",
        which_model="denoising",
        cls_free_guidance=cls_free_guidance,
        sampler_name=sampler_name,
        do_purity_sampling=do_purity_sampling,
        dt=dt,
        use_tag=use_tag,
        discrete_time=discrete_time,
        num_timesteps=num_timesteps,
    )
    # Save the total number of samples drawn to the config file
    eval_cfg.sampler.num_samples = num_samples
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
    loaded_state = torch.load(
        eval_cfg.denoising_model_checkpoint_path, map_location=device
    )
    denoising_model.load_state_dict(loaded_state["model"])

    # Load clean classifier for evaluation
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

    if not cls_free_guidance:
        # Load noisy classifier for PG
        cls_model_train_cfg = bookkeeping.load_ml_collections(
            Path(eval_cfg.cls_model_train_config_path)
        )
        cls_model = CNNModel(cls_model_train_cfg).to(device)
        loaded_state = torch.load(
            eval_cfg.cls_model_checkpoint_path, map_location=device
        )
        cls_model.load_state_dict(loaded_state["model"])
        print(f"Using noisy classifier from {eval_cfg.cls_model_checkpoint_path}")
    else:
        # For PFG we don't need the noisy classifier
        cls_model = None

    # Parse sampling configs
    S = eval_cfg.data.S
    D = eval_cfg.data.shape
    stochasticity = eval_cfg.sampler.noise
    x1_temp = eval_cfg.sampler.x1_temp
    purity_temp = eval_cfg.sampler.purity_temp
    do_purity_sampling = eval_cfg.sampler.do_purity_sampling
    argmax_final = eval_cfg.sampler.argmax_final
    max_t = eval_cfg.sampler.max_t
    cls_free_guidance = eval_cfg.model.cls_free_guidance
    use_tag = eval_cfg.sampler.use_tag
    batch_size = eval_cfg.sampler.batch_size
    mask_idx = S - 1
    num_timesteps = eval_cfg.num_timesteps

    # Log configs
    config_str = json.dumps(eval_cfg.to_dict(), indent=2)
    config_str = config_str.replace("\n", "\n\n")
    writer.add_text("Config", config_str)

    # Generate samples
    utils.set_random_seed(eval_cfg.seed)
    all_metrics = []
    # Iterate over sampling replicates
    for n in range(num_replicates):
        # Iterate over target classes
        for c in eval_cfg.target_classes:
            # Define the relevant partial functions
            if not cls_free_guidance:
                # Predictor guidance
                denoising_model_func = lambda xt, t: denoising_model((xt, None), t)
                predictor_log_prob = lambda xt, t: utils.get_cls_log_probs(
                    cls_model,
                    xt,
                    t,
                    y=c * torch.ones((t.shape[0],), dtype=torch.long, device=t.device),
                )
                cond_denoising_model_func = None
            else:
                # Predictor-free guidance
                # The "null" class representing unconditional training
                null_class = denoising_model.num_cls
                denoising_model_func = lambda xt, t: denoising_model(
                    (
                        xt,
                        null_class
                        * torch.ones((t.shape[0],), dtype=torch.long, device=t.device),
                    ),
                    t,
                )
                predictor_log_prob = None
                # Conditional denoising model is the denoising model with class input
                cond_denoising_model_func = lambda xt, t: denoising_model(
                    (
                        xt,
                        c
                        * torch.ones((t.shape[0],), dtype=torch.long, device=t.device),
                    ),
                    t,
                )

            for i, guide_temp in enumerate(eval_cfg.guide_temps):
                # Iterate over guidance temperatures
                start_time = time.time()
                if discrete_time:
                    samples = digress_utils.d3pm_sampling(
                        num_samples,
                        denoising_model_func,
                        S=S,
                        D=D,
                        device=device,
                        timesteps=num_timesteps,
                        mask_idx=mask_idx,
                        batch_size=batch_size,
                        predictor_log_prob=predictor_log_prob,
                        guide_temp=guide_temp,
                        x1_temp=x1_temp,
                    )
                else:
                    samples = fm_utils.flow_matching_sampling(
                        num_samples,
                        denoising_model_func,
                        S=S,
                        D=D,
                        device=device,
                        dt=dt,
                        mask_idx=mask_idx,
                        batch_size=batch_size,
                        predictor_log_prob=predictor_log_prob,
                        cond_denoising_model=cond_denoising_model_func,
                        guide_temp=guide_temp,
                        use_tag=use_tag,
                        stochasticity=stochasticity,
                        argmax_final=argmax_final,
                        max_t=max_t,
                        x1_temp=x1_temp,
                        do_purity_sampling=do_purity_sampling,
                        purity_temp=purity_temp,
                    )
                end_time = time.time()
                print(f"Sampling elapsed time: {end_time - start_time:.6f} seconds")

                # Save samples to npy file
                if not cls_free_guidance:
                    name = f"class_{c}-temp_{guide_temp}-grad_{use_tag}-rep_{n}.npy"
                else:
                    name = f"class_{c}-temp_{guide_temp}-rep_{n}.npy"
                with open(os.path.join(save_dir, name), "wb") as f:
                    np.save(f, samples)

                # Save metrics to pkl file
                metrics = utils.print_eval_metrics(
                    samples, cls_clean_model, eval_cfg, target_class=c
                )
                # Plot conditional fbd and target class prob against temperature
                writer.add_scalar(f"class_{c}/fbd_cond", metrics["fbd_cond"], i)
                writer.add_scalar(
                    f"class_{c}/target_cls_prob", metrics["target_cls_prob"], i
                )
                writer.add_scalar(
                    f"class_{c}/target_cls_frac", metrics["target_cls_frac"], i
                )
                writer.add_scalar(f"class_{c}/guide_temp", guide_temp, i)
                all_metrics.append(
                    (
                        n,
                        c,
                        guide_temp,
                        metrics["fbd_cond"],
                        metrics["target_cls_prob"],
                        metrics["target_cls_frac"],
                    )
                )
                with open(os.path.join(save_dir, "metrics.pkl"), "wb") as f:
                    pickle.dump(all_metrics, f)


if __name__ == "__main__":
    main()
