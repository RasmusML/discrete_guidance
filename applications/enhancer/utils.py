import torch
import torch.nn.functional as F
import numpy as np
import random
from scipy.linalg import sqrtm
from collections import defaultdict
from tqdm import tqdm
import torchmetrics

import src.fm_utils as fm_utils
import src.digress_utils as digress_utils
import applications.bookkeeping as bookkeeping
from applications.enhancer.dataset import EnhancerDataset, GeneratedDataset


def set_random_seed(random_seed):
    """Set random seed(s) for reproducibility."""
    # Set random seeds for any modules that potentially use randomness
    random.seed(random_seed)
    np.random.seed(random_seed + 1)
    torch.random.manual_seed(random_seed + 2)


def get_cls_log_probs(cls_model, xt, t, y, batch_size=1000):
    """
    Compute classifier log probs
    """
    B = xt.shape[0]
    if B <= batch_size:
        cls_log_probs = -F.cross_entropy(cls_model((xt, None), t), y, reduction="none")
    else:
        xt_chunks = torch.split(xt, batch_size)
        t_chunks = torch.split(t, batch_size)
        y_chunks = torch.split(y, batch_size)
        cls_log_probs = []
        for i in range(len(xt_chunks)):
            cls_log_probs.append(
                -F.cross_entropy(
                    cls_model((xt_chunks[i], None), t_chunks[i]),
                    y_chunks[i],
                    reduction="none",
                )
            )
        cls_log_probs = torch.cat(cls_log_probs, dim=0)
    return cls_log_probs


def cls_clean_loss(model, batch_data, return_logits=False):
    """
    Compute CE loss for a unnoised/clean classifier
    """
    x1, cls = batch_data
    # The model outputs logits over number of classes
    logits = model((x1, None), None)  # (B, #classes)
    loss = F.cross_entropy(logits, cls, reduction="mean")
    if not return_logits:
        return loss
    else:
        return loss, logits


def cls_noisy_loss_masking(model, batch_data, cfg, S=5, return_logits=False):
    """
    Compute CE loss for noisy classifier

    Note: we are not using the `predictor_loss_masking` function directly
    from fm_utils because we want to have the option of returning logits
    """
    x1, cls = batch_data
    B, D = x1.shape
    # <mask> is the last index
    mask_idx = S - 1

    # Sample xt depending on whether discrete or continuous time is used
    if cfg.discrete_time:
        t = (
            torch.randint(low=1, high=cfg.num_timesteps + 1, size=(B,))
            .float()
            .to(x1.device)
        )
        xt = digress_utils.d3pm_sample_xt(x1, t, mask_idx, cfg.num_timesteps)
    else:
        t = torch.rand((B,)).to(x1.device)
        xt = fm_utils.sample_xt(x1, t, mask_idx)

    # The model outputs logits over number of classes
    logits = model((xt, cls), t)  # (B, #classes)
    loss = F.cross_entropy(logits, cls, reduction="mean")
    if not return_logits:
        return loss
    else:
        return loss, logits


def denoising_model_loss(model, batch_data, cfg):
    """
    Thin wrapper around fm_utils.flow_matching_loss_masking to accommodate
    the the enhancer example
    Allows for both unconditional training and training with PFG
    Used in both training and evaluation

    """
    cls_free_guidance = cfg.model.cls_free_guidance
    cls_free_noclass_ratio = cfg.model.cls_free_noclass_ratio
    mask_idx = cfg.data.S - 1  # <mask> is the last index

    x1, cls = batch_data
    B, D = x1.shape

    # Predictor-free guidance model uses the class information as input
    if cls_free_guidance:
        # Set `cls_free_noclass_ratio` fraction of the classes to
        # the unconditional class when training
        cls_input = torch.where(
            torch.rand(B, device=x1.device) >= cls_free_noclass_ratio,
            cls.squeeze(),
            model.num_cls,
        )
    else:
        # Unconditional model doesn't use class information
        cls_input = None

    # Define partial function for denoising model
    model_func = lambda xt, t: model((xt, cls_input), t)
    # Compute loss on batch
    if cfg.discrete_time:
        print("discrete time")
        batch_loss = digress_utils.d3pm_loss_masking(
            denoising_model=model_func,
            x1=x1,
            mask_idx=mask_idx,
            timesteps=cfg.num_timesteps,
        )
    else:
        batch_loss = fm_utils.flow_matching_loss_masking(
            denoising_model=model_func, x1=x1, mask_idx=mask_idx
        )
    return batch_loss


def train_on_batch(
    batch_data, model, optimizer, training_state, cfg, which_model="denoising"
):
    """
    Train the model on the batch (i.e. update the model parameters) and
    return the loss of the batch data.
    """
    model.train()

    # Parse training arguments
    clip_grad = cfg.training.clip_grad
    warmup = cfg.training.warmup
    lr = cfg.training.lr

    optimizer.zero_grad()

    # Compute loss for the model
    # Depending on whether discrete or continuous time is used
    # the loss would be different
    if which_model == "denoising":
        batch_loss = denoising_model_loss(model, batch_data, cfg)
    elif which_model == "cls_noisy":
        batch_loss = cls_noisy_loss_masking(model, batch_data, cfg)
    elif which_model == "cls_clean":
        batch_loss = cls_clean_loss(model, batch_data)

    batch_loss.backward()

    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if warmup > 0:
        for g in optimizer.param_groups:
            g["lr"] = lr * np.minimum(training_state["n_iter"] / warmup, 1.0)

    optimizer.step()

    training_state["n_iter"] += 1
    training_state["loss"] = batch_loss.item()
    return training_state


@torch.no_grad
def eval_on_batch(
    batch_data, model, cfg, which_model="denoising", cls_clean_model=None
):
    """
    Return evaluation loss
    If clean classifier available and model is denoising, then also return
    embeddings of the batch for FBD computation
    If so, eval_cfg needs to be provided for configuring sampling
    """
    model.eval()
    eval_output = dict()

    # First evaluate the clean classifier if available
    if cls_clean_model is not None:
        # Get embeddings on the batch
        _, embeds_data = cls_clean_model(batch_data, t=None, return_embedding=True)
        embeds_data = embeds_data.detach().cpu().numpy()
        eval_output["embeds_data"] = embeds_data

    if which_model == "denoising":
        loss = denoising_model_loss(model, batch_data, cfg)
    elif which_model == "cls_noisy":
        loss, logits = cls_noisy_loss_masking(
            model, batch_data, cfg, return_logits=True
        )
        eval_output["logits"] = logits
    elif which_model == "cls_clean":
        loss, logits = cls_clean_loss(model, batch_data, return_logits=True)
        eval_output["logits"] = logits

    eval_output["loss"] = loss.item()
    return eval_output


def train_model_on_dataset(
    cfg,
    train_dataloader,
    model,
    optimizer,
    writer=None,
    which_model="denoising",
    valid_dataloader=None,
    checkpoint_dir=None,
    cls_clean_model=None,
    eval_cfg=None,
):
    if writer is None:
        writer = bookkeeping.DummyWriter("none")
    torch.autograd.set_detect_anomaly(True)
    device = cfg.device

    # Number of epochs is the total number of iterations divided by number of batches
    num_epochs = cfg.training.n_iters // len(train_dataloader)
    print(
        f"Training the '{which_model}' model for '{num_epochs}' epochs on device '{device}'."
    )

    if which_model != "denoising":
        # If training classifier, evaluate accuracy for early stopping
        # auroc_metric = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes, average=None).to(device)
        accuracy_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=cfg.data.num_classes, average=None
        ).to(device)
        auprc_metric = torchmetrics.classification.MulticlassAveragePrecision(
            num_classes=cfg.data.num_classes, average=None
        ).to(device)

    # Loop over the training epochs
    epoch_train_output = defaultdict(list)
    epoch_valid_output = defaultdict(list)
    # Number of training steps is aggregated over all epochs and minibatchs
    training_state = dict(n_iter=0)

    # Keep track of the best validation loss to save checkpoint
    best_valid_loss = np.inf

    if cls_clean_model is not None:
        # Also save ckpt based on fbd
        best_fbd = np.inf
        # Computing fbd requires sampling, so needs to parse the sampling configs
        if eval_cfg is None:
            raise ValueError(
                "Clean classifier is provided for FBD calculation, but eval_cfg is not"
            )
        S = eval_cfg.data.S
        D = eval_cfg.data.shape
        dt = eval_cfg.sampler.dt
        stochasticity = eval_cfg.sampler.noise
        x1_temp = eval_cfg.sampler.x1_temp
        purity_temp = eval_cfg.sampler.purity_temp
        do_purity_sampling = eval_cfg.sampler.do_purity_sampling
        argmax_final = eval_cfg.sampler.argmax_final
        max_t = eval_cfg.sampler.max_t
        batch_size = eval_cfg.sampler.batch_size
        mask_idx = S - 1

    if which_model != "denoising":
        best_accuracy = 0.0
        best_auprc = 0.0

    for epoch in range(num_epochs):
        batch_train_output = defaultdict(list)
        pbar = tqdm(train_dataloader)
        for batch_train_data in pbar:
            # Train on the batch for the specified model
            training_state = train_on_batch(
                batch_train_data,
                model,
                optimizer,
                training_state,
                cfg,
                which_model=which_model,
            )
            for k, v in training_state.items():
                if k == "n_iter":
                    continue
                batch_train_output[k].append(v)
                # Add per minibatch output
                writer.add_scalar(f"{k}/train/batch", v, training_state["n_iter"])
            pbar.set_description(f"train loss={training_state['loss']:.4e}")

            # Log training checkpoint for every fixed number of iterations
            if (
                training_state["n_iter"] % cfg.saving.checkpoint_freq == 0
                or training_state["n_iter"] == cfg.training.n_iters - 1
            ):
                if checkpoint_dir:
                    ckpt_state = dict(
                        model=model,
                        optimizer=optimizer,
                        n_iter=training_state["n_iter"],
                        epoch=epoch,
                    )
                    bookkeeping.save_checkpoint(
                        checkpoint_dir, ckpt_state, cfg.saving.num_checkpoints_to_keep
                    )

            # Log archive checkpoint for larger interval
            if (
                training_state["n_iter"] % cfg.saving.checkpoint_archive_freq == 0
                and checkpoint_dir
            ):
                ckpt_state = dict(
                    model=model,
                    optimizer=optimizer,
                    n_iter=training_state["n_iter"],
                    epoch=epoch,
                )
                bookkeeping.save_archive_checkpoint(
                    checkpoint_dir, ckpt_state, f"ckpt-epoch_{epoch}.pt"
                )

        for k, v in batch_train_output.items():
            avg = np.mean(np.array(v))
            epoch_train_output[k].append(avg)
            writer.add_scalar(f"{k}/train/avg", avg, epoch)

        if valid_dataloader is not None:
            batch_valid_output = defaultdict(list)
            for batch_valid_data in valid_dataloader:
                # Optionally pass in the clean classifier for computing the embeddings
                # fo FBD calculation
                eval_output = eval_on_batch(
                    batch_valid_data,
                    model,
                    cfg,
                    which_model,
                    cls_clean_model=cls_clean_model,
                )
                for k, v in eval_output.items():
                    if k != "logits":
                        batch_valid_output[k].append(v)
                    else:
                        # Add probs and labels for overall metrics computation
                        probs = torch.nn.functional.softmax(v, dim=1)
                        labels = batch_valid_data[1]
                        accuracy_metric(probs, labels)
                        auprc_metric(probs, labels)

                perplexity = np.exp(eval_output["loss"])
                batch_valid_output["perplexity"].append(perplexity)


            ## Log metrics for the epoch
            for k, v in batch_valid_output.items():
                if k not in ["embeds_data", "logits"]:
                    avg = np.mean(np.array(v))
                    epoch_valid_output[k].append(avg)
                    writer.add_scalar(f"{k}/valid", avg, epoch)

                print(f"Validation {k}: {avg:.4f}")

            # Additional metrics for clean classifier
            if which_model != "denoising":
                # Mean accuracy over classes
                accuracies = accuracy_metric.compute().cpu().numpy()
                accuracy_metric.reset()
                acc = accuracies.mean()
                epoch_valid_output["accuracy"].append(acc)
                writer.add_scalar(f"accuracy/valid", acc, epoch)
                # Mean auprc over classes
                auprcs = auprc_metric.compute().cpu().numpy()
                auprc_metric.reset()
                auprc = auprcs[~np.isnan(auprcs)].mean()
                epoch_valid_output["auprc"].append(auprc)
                writer.add_scalar(f"auprc/valid", auprc, epoch)

            # Compute FBD if embeddings available
            if "embeds_data" in batch_valid_output:
                # Concatenate the embeddings of the data over all batches
                embeds_data = np.concatenate(batch_valid_output["embeds_data"])
                # Draw unconditional samples
                # Note: this will slow down training substantially
                # For approximation, instead of drawing same number of samples at val set
                # we draw 2000 samples
                # For predictor-free guidance, ideally one should sample conditionally
                # by class proportion instead of using unconditional samples
                # We currently don't implement this
                denoising_model_func = lambda xt, t: model((xt, None), t)
                samples = fm_utils.flow_matching_sampling(
                    num_samples=2000,
                    denoising_model=denoising_model_func,
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
                )
                samples = torch.from_numpy(samples).long().to(device)
                _, embeds_samples = cls_clean_model(
                    (samples, None), t=None, return_embedding=True
                )
                embeds_samples = embeds_samples.detach().cpu().numpy()
                fbd = get_wasserstein_dist(embeds_data, embeds_samples)
                epoch_valid_output["fbd"].append(fbd)
                writer.add_scalar(f"fbd/valid", fbd, epoch)

            ## Checkpoints for early stopping
            # Save archive checkpoint for model with best validation loss
            epoch_valid_loss = epoch_valid_output["loss"][-1]
            if epoch_valid_loss < best_valid_loss and checkpoint_dir:
                best_valid_loss = epoch_valid_loss
                ckpt_state = dict(
                    model=model,
                    optimizer=optimizer,
                    n_iter=training_state["n_iter"],
                    epoch=epoch,
                    valid_loss=epoch_valid_loss,
                )
                bookkeeping.save_archive_checkpoint(
                    checkpoint_dir, ckpt_state, "ckpt_best_val_loss.pt"
                )
                print(
                    f"New best validation loss at epoch {epoch}: {best_valid_loss:.4f}, saved checkpoint"
                )

            # For clean classifier, save checkpoint with best accuracy or auprc
            if "accuracy" in epoch_valid_output and checkpoint_dir:
                epoch_accuracy = epoch_valid_output["accuracy"][-1]
                epoch_auprc = epoch_valid_output["auprc"][-1]
                ckpt_state = dict(
                    model=model,
                    optimizer=optimizer,
                    n_iter=training_state["n_iter"],
                    epoch=epoch,
                    valid_loss=epoch_valid_loss,
                    accuracy=epoch_accuracy,
                    auprc=epoch_auprc,
                )
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    bookkeeping.save_archive_checkpoint(
                        checkpoint_dir, ckpt_state, "ckpt_best_accuracy.pt"
                    )
                    print(
                        f"New best accuracy at epoch {epoch}: {best_accuracy:.4f}, saved checkpoint"
                    )
                if epoch_auprc > best_auprc:
                    best_auprc = epoch_auprc
                    bookkeeping.save_archive_checkpoint(
                        checkpoint_dir, ckpt_state, "ckpt_best_auprc.pt"
                    )
                    print(
                        f"New best auprc at epoch {epoch}: {best_auprc:.4f}, saved checkpoint"
                    )

            # If available, save checkpoint based on fbd
            if "fbd" in epoch_valid_output:
                epoch_fbd = epoch_valid_output["fbd"][-1]
                if epoch_fbd < best_fbd and checkpoint_dir:
                    best_fbd = epoch_fbd
                    ckpt_state = dict(
                        model=model,
                        optimizer=optimizer,
                        n_iter=training_state["n_iter"],
                        epoch=epoch,
                        valid_loss=epoch_valid_loss,
                        fbd=epoch_fbd,
                    )
                    bookkeeping.save_archive_checkpoint(
                        checkpoint_dir, ckpt_state, f"ckpt_best_fbd.pt"
                    )
                    print(
                        f"New best fbd at epoch {epoch}: {best_fbd:.4f}, saved checkpoint"
                    )


def get_labels_from_cls_labeler_model(samples, cls_labeler_model, batch_size=500):
    """
    Assign class labels to generated samples using a trained classifer

    Args:
        samples (np.array): Shape (N, D)
    Returns:
        cls_labels (np.array): Shape (N,)
    """
    cls_labeler_model.eval()

    device = next(cls_labeler_model.parameters()).device
    gen_loader = torch.utils.data.DataLoader(
        GeneratedDataset(x=samples, device=device, y=None),
        batch_size=batch_size,
        shuffle=False,
    )

    # Get log probs of the target class for all the samples
    # Take argmax as labels
    cls_labels = []
    for batch in gen_loader:
        logits = cls_labeler_model(batch, t=None)  # (B, #classes)
        labels = logits.argmax(dim=1).detach().cpu().numpy()  # (B,)
        cls_labels.append(labels)
    cls_labels = np.concatenate(cls_labels, axis=0)
    return cls_labels


def print_eval_metrics(
    samples, cls_clean_model, eval_cfg, target_class=None, split="train"
):
    fbd_uncond = calc_fbd(cls_clean_model, samples, eval_cfg, which_data=split)
    if target_class is not None:
        fbd_cond = calc_fbd(
            cls_clean_model,
            samples,
            eval_cfg,
            target_class=target_class,
            which_data=split,
        )
        target_cls_prob, target_cls_frac = calc_target_class_prob(
            cls_clean_model, samples, target_class
        )
        print(
            f"FBD(uncond): {fbd_uncond:.3f}, FBD(cls={target_class}): {fbd_cond:.3f}, \
            prob(cls={target_class}): {target_cls_prob:.3f}, \
            frac(argmax_cls={target_class}): {target_cls_frac:.3f}"
        )
        return dict(
            fbd_uncond=fbd_uncond,
            fbd_cond=fbd_cond,
            target_cls_prob=target_cls_prob,
            target_cls_frac=target_cls_frac,
        )
    else:
        print(f"FBD(uncond): {fbd_uncond:.3f}")
        return dict(fbd_uncond=fbd_uncond)


def average_pairwise_hamming_distance(seqs):
    # Step 1: Compute pairwise differences using broadcasting
    diffs = np.expand_dims(seqs, 1) != np.expand_dims(seqs, 0)

    # Step 2: Sum the differences across the sequence dimension to get Hamming distances
    hamming_distances = np.sum(diffs, axis=-1)

    # Step 3: Since the matrix is symmetric and has zeros on the diagonal,
    # we only need to average the upper triangular part (excluding the diagonal).
    N = seqs.shape[0]
    mean_dist = np.sum(hamming_distances) / (N * (N - 1))
    return mean_dist


def get_wasserstein_dist(embeds1, embeds2):
    if (
        np.isnan(embeds2).any()
        or np.isnan(embeds1).any()
        or len(embeds1) == 0
        or len(embeds2) == 0
    ):
        return float("nan")
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist


def convert_seqs_to_array(seqs):
    order = {"A": 0, "C": 1, "G": 2, "T": 3}
    arrays = []
    for seq in seqs:
        arrays.append([order[char] for char in seq])
    return np.vstack(arrays)


def calc_target_class_prob(
    cls_model, generated_samples, target_class, batch_size=500, return_std=False
):
    """
    Calculate two metrics of the generated samples for the target class under
    the oracle classifier, where y* is the target class and x are
    generated sequences for that class (either with PFG or PG)

    1. E_x[p(y=y* | x)], the average target class probability
    2. E_x[I{y* = argmax_{y} p(y | x)}], the fraction of times when
        the target class is the most likely class
    """
    cls_model.eval()

    device = next(cls_model.parameters()).device
    y = torch.full((generated_samples.shape[0],), target_class, dtype=int)
    gen_loader = torch.utils.data.DataLoader(
        GeneratedDataset(x=generated_samples, device=device, y=y),
        batch_size=batch_size,
        shuffle=False,
    )

    # Get log probs of the target class for all the samples
    cls_probs, cls_fracs = [], []
    for batch in gen_loader:
        logits, _ = cls_model(
            batch,
            t=torch.ones((batch[0].shape[0],), device=device),
            return_embedding=True,
        )
        probs = F.softmax(logits, dim=-1)  # (B, #classes)

        # Calculate the target class probability
        cls_prob = probs[:, target_class].detach().cpu().numpy()  # (B,)
        cls_probs.append(cls_prob)

        # Calculate the fraction of times target class is most likely
        cls_frac = (probs.argmax(dim=1) == target_class).detach().cpu().numpy()  # (B,)
        cls_fracs.append(cls_frac)

    cls_probs = np.concatenate(cls_probs, axis=0)
    cls_fracs = np.concatenate(cls_fracs, axis=0)
    cls_probs_mean = cls_probs.mean()
    cls_fracs_mean = cls_fracs.mean()
    if return_std:
        cls_probs_std = cls_probs.std()
        cls_fracs_std = cls_fracs.std()
        return cls_probs_mean, cls_fracs_mean, cls_probs_std, cls_fracs_std
    else:
        return cls_probs_mean, cls_fracs_mean


def calc_fbd(
    cls_model,
    generated_samples,
    eval_cfg,
    which_data="train",
    batch_size=500,
    target_class=None,
    return_embed=False,
):
    """
    Calculate FBD of generated samples with the dataset
    cls_model should be the clean classifier in most cases, but this can
    also be used with a noisy model

    Args:
        generated_samples (torch.tensor): Shape (B, D)
        which_data: Either train, val or test
    """
    cls_model.eval()
    device = eval_cfg.device
    data_loader = torch.utils.data.DataLoader(
        EnhancerDataset(eval_cfg, which_data), batch_size=batch_size, shuffle=False
    )
    gen_loader = torch.utils.data.DataLoader(
        GeneratedDataset(x=generated_samples, device=device, y=None),
        batch_size=batch_size,
        shuffle=False,
    )

    # Get embeddings for the data
    embeds_data = []
    for batch in data_loader:
        t = torch.ones((batch[0].shape[0],), device=device)
        _, embeds = cls_model(batch, t=t, return_embedding=True)

        if target_class is not None:
            # Only get the embedding for the target class
            cls = batch[1]
            embeds = embeds[cls == target_class]

        embeds = embeds.detach().cpu().numpy()
        embeds_data.append(embeds)
    embeds_data = np.concatenate(embeds_data, axis=0)

    # Get embeddings for the samples
    embeds_generated = []
    for batch in gen_loader:
        t = torch.ones((batch[0].shape[0],), device=device)
        _, embeds = cls_model(batch, t=t, return_embedding=True)
        embeds = embeds.detach().cpu().numpy()
        embeds_generated.append(embeds)
    embeds_generated = np.concatenate(embeds_generated, axis=0)

    fbd = get_wasserstein_dist(embeds_data, embeds_generated)
    if not return_embed:
        return fbd
    else:
        return fbd, (embeds_data, embeds_generated)
