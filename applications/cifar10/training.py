import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import defaultdict
import torchmetrics

import applications.bookkeeping as bookkeeping


def train_manager_on_cifar(
        diffusion_manager, train_dataloader, cfg, writer=None, 
        which_model='denoising', valid_dataloader=None, checkpoint_dir=None):
    """
    Train the denoising model or the property model (with noise) 
    under the diffusion manager.
    
    Args:
        diffusion_manager (managers.base_diffusion_manager.BaseDiffusionManager):
            Diffusion manager that contains the diffusion-model and abstracts its 
            functionality. Also contains property-model used for guided diffusion.
        train_dataloader (torch.utils.data.Dataloader): Dataloader for the 
            iteration over batches of the training dataset.
        which_model (str): Which model to train 'denoising' or 'property'.
        valid_dataloader (torch.utils.data.Dataloader or None): If not None,
            dataloader for the iteration over batches of the validation 
            dataset.
            (Default: None)
        num_epochs (int): Number of epochs to train for.
            (Default: 10)
        random_seed(int): Random seed to be used for training.
            (Default: 42)
        display_info_every_nth_epoch (int): Display information every nth epoch.
            (Default: 1)
        plot_training_curves (bool): Should the training curves (e.g. loss vs. epoch)
            be plotted or not?
            (Default: False)
    
    Return:
        None
    
    """
    if writer is None:
        writer = bookkeeping.DummyWriter('none')
    torch.autograd.set_detect_anomaly(True)
    device = cfg.device
    
    # Number of epochs is the total number of iterations divided by number of batches
    num_epochs = cfg.training.n_iters // len(train_dataloader)
    print(f"Training the '{which_model}' model for '{num_epochs}' epochs on device '{diffusion_manager.device}'.")

    if which_model != 'denoising':
        # If training classifier, evaluate accuracy for early stopping
        # auroc_metric = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes, average=None).to(device)
        accuracy_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=cfg.model.num_classes, average=None).to(device)

    # Loop over the training epochs
    epoch_train_output = defaultdict(list)
    epoch_valid_output = defaultdict(list)
    # Number of training steps is aggregated over all epochs and minibatchs
    training_state = dict(n_iter=0)
    # Keep track of the best validation loss to save checkpoint
    best_valid_loss = np.inf
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        batch_train_output = defaultdict(list)
        pbar = tqdm(train_dataloader)
        for batch_train_data in pbar:
            # Train on the batch for the specified model
            training_state = diffusion_manager.train_on_batch(
                batch_train_data, 
                which_model, 
                training_state,
                train_dataloader.batch_size,
            )
            for k, v in training_state.items():
                if k == 'n_iter':
                    continue
                batch_train_output[k].append(v)
                # Add per minibatch output
                writer.add_scalar(f'{k}/train/batch', v, training_state['n_iter'])
            pbar.set_description(f"train loss={training_state['loss']:.4e}")

            if which_model == 'denoising':
                # Log denoised images
                if training_state['n_iter'] % cfg.saving.log_low_freq == 0:
                    fig = denoising_images(diffusion_manager, batch_train_data)
                    writer.add_figure('images/train', fig, training_state['n_iter'])

            # Log training checkpoint for fixed iterations
            if training_state['n_iter'] % cfg.saving.checkpoint_freq == 0 or training_state['n_iter'] == cfg.training.n_iters-1:
                if checkpoint_dir:
                    ckpt_state = dict(
                        model=diffusion_manager.model_dict[which_model], 
                        optimizer=diffusion_manager.optim_dict[which_model],
                        n_iter=training_state['n_iter'],
                        epoch=epoch
                    )
                    bookkeeping.save_checkpoint(
                        checkpoint_dir, ckpt_state,
                        cfg.saving.num_checkpoints_to_keep)

        for k, v in batch_train_output.items():
            avg = np.mean(np.array(v))
            epoch_train_output[k].append(avg)
            writer.add_scalar(f'{k}/train/avg', avg, epoch)

        if valid_dataloader is not None:
            batch_valid_output = defaultdict(list)
            for batch_valid_data in valid_dataloader:
                eval_output = diffusion_manager.eval_loss_on_batch(
                    batch_valid_data, which_model, valid_dataloader.batch_size,
                    return_logits=(which_model == 'property')
                )
                for k, v in eval_output.items():
                    if k != 'logits':
                        batch_valid_output[k].append(v)
                    else:
                        # Add probs and labels for overall metrics computation
                        probs = torch.nn.functional.softmax(v, dim=1)
                        labels = batch_valid_data['y']
                        accuracy_metric(probs, labels)

            ## Log metrics for the epoch
            for k, v in batch_valid_output.items():
                avg = np.mean(np.array(v))
                epoch_valid_output[k].append(avg)
                writer.add_scalar(f'{k}/valid', avg, epoch)

            if which_model == 'denoising':
                fig = denoising_images(diffusion_manager, batch_valid_data)
                writer.add_figure('images/valid', fig, epoch)
            else:
                accuracies = accuracy_metric.compute().cpu().numpy()
                accuracy_metric.reset()
                # Mean accuracy over classes
                acc = accuracies.mean()
                epoch_valid_output['accuracy'].append(acc)
                writer.add_scalar(f'accuracy/valid', acc, epoch)

            ## Checkpoints for early stopping
            # Save archive checkpoint for model with best validation loss
            epoch_valid_loss = epoch_valid_output['loss'][-1]
            if epoch_valid_loss < best_valid_loss and checkpoint_dir:
                best_valid_loss = epoch_valid_loss
                ckpt_state = dict(
                    model=diffusion_manager.model_dict[which_model], 
                    optimizer=diffusion_manager.optim_dict[which_model],
                    n_iter=training_state['n_iter'],
                    epoch=epoch,
                    valid_loss=epoch_valid_loss,
                )
                bookkeeping.save_archive_checkpoint(
                    checkpoint_dir, ckpt_state, 'ckpt_best_val_loss.pt')
                print(f'New best validation loss at epoch {epoch}: {best_valid_loss:.4f}, saved checkpoint')

            # For clean classifier, save checkpoint with best accuracy
            if 'accuracy' in epoch_valid_output:
                epoch_accuracy = epoch_valid_output['accuracy'][-1]
                if epoch_accuracy > best_accuracy and checkpoint_dir:
                    best_accuracy = epoch_accuracy
                    ckpt_state = dict(
                        model=diffusion_manager.model_dict[which_model], 
                        optimizer=diffusion_manager.optim_dict[which_model],
                        n_iter=training_state['n_iter'],
                        epoch=epoch,
                        valid_loss=epoch_valid_loss,
                        accuracy=epoch_accuracy
                    )
                    bookkeeping.save_archive_checkpoint(
                        checkpoint_dir, ckpt_state, f'ckpt_best_accuracy.pt')
                    print(f'New best accuracy at epoch {epoch}: {best_accuracy:.4f}, saved checkpoint')


def imgtrans(x):
    if isinstance(x, torch.Tensor):
        # C,H,W -> H,W,C
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
    elif isinstance(x, np.ndarray):
        x = x.transpose(1, 2, 0)
    else:
        raise TypeError
    return x


def denoising_images(diffusion_manager, batch):
    cfg = diffusion_manager.cfg
    ts = [0.01, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
    C,H,W = cfg.data.shape
    B = 1
    S = cfg.data.S
    x = batch['x']

    fig, ax = plt.subplots(6, len(ts))
    for img_idx in range(3):
        for t_idx in range(len(ts)):
            qt0 = diffusion_manager.forward_process.get_Q_t(torch.tensor([ts[t_idx]], device=diffusion_manager.device)) # (B, S, S)
            qt0_rows = qt0[
                0, x[img_idx, :].flatten().long(), :
            ]
            x_t_cat = torch.distributions.categorical.Categorical(
                qt0_rows
            )
            x_t = x_t_cat.sample().view(1, C*H*W)

            x_0_logits = diffusion_manager.model_dict['denoising']({'x': x_t}, torch.tensor([ts[t_idx]], device=diffusion_manager.device)).view(B,C,H,W,S)
            x_0_max_logits = torch.max(x_0_logits, dim=4)[1]

            ax[2*img_idx, t_idx].imshow(imgtrans(x_t.view(B,C,H,W)[0, ...].detach().cpu()))
            ax[2*img_idx, t_idx].axis('off')
            ax[2*img_idx+1, t_idx].imshow(imgtrans(x_0_max_logits[0, ...].detach().cpu()))
            ax[2*img_idx+1, t_idx].axis('off')
    return fig


def train_single_epoch(
        model, train_loader, loss_fn, optimizer, scheduler, 
        training_state, writer, clip_grad=1.0):
    """
    General purpose util function to train a model 
    For the purpose of cifar, probably a classifier that is not a noisy classifier
    (noisy classifier training is handled in base_diffusion_manager)
    for one epoch over all minibatches

    Args:
        model (torch.nn.Module)
        training_state (dict): Holding the current status of training, including:
            loss: The current training loss
            n_iter: The total number of iterations (number of gradient steps)

    Returns:
        batch_train_output (defaultdict(list)): 
            For each field in training_state, store a list of values, one per
            minibatch
        training_state (dict): The updated training state at the end of this epoch 
    """
    model.train()
    pbar = tqdm(train_loader)
    batch_train_output = defaultdict(list)
    for batch in pbar:
        if isinstance(batch, dict):
            input, label = batch['x'], batch['y']
        elif isinstance(batch, tuple) or isinstance(batch, list):
            assert (len(batch) == 2)
            input, label = batch
        else:
            raise ValueError(f'batch should be dict or tuple, not {type(batch)}')
        input = input.contiguous().cuda()
        label = label.contiguous().cuda()
        pred = model(input)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        # Update training state
        training_state['loss'] = loss.item()
        training_state['n_iter'] += 1

        for k, v in training_state.items():
            if k == 'n_iter':
                continue
            batch_train_output[k].append(v)
            # Add per minibatch output
            writer.add_scalar(f'{k}/train/batch', v, training_state['n_iter'])
        pbar.set_description(f"train loss={training_state['loss']:.4e}")

    return batch_train_output, training_state


def eval_on_data(model, data_loader, loss_fn, metrics):
    """
    Return evaluation metrics for a model over the dataset
    """
    model.eval()
    total_loss = 0
    pred_all, labels_all = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                input, label = batch['x'], batch['y']
            elif isinstance(batch, tuple) or isinstance(batch, list):
                assert (len(batch) == 2)
                input, label = batch
            else:
                raise ValueError(f'batch should be dict or tuple, not {type(batch)}')
            input = input.contiguous().cuda()
            label = label.contiguous().cuda()
            logits = model(input)
            loss = loss_fn(logits, label)
            total_loss += loss * logits.shape[0]
            if len(metrics) > 0:
                pred = logits.argmax(dim=1)
                pred_all.append(pred.detach().cpu().numpy().flatten())
                labels_all.append(label.detach().cpu().numpy().flatten())
        avg_loss = total_loss / len(data_loader.dataset)

    if len(pred_all) > 0:
        pred_all = np.concatenate(pred_all)
        labels_all = np.concatenate(labels_all)
    outputs = dict(loss=avg_loss.item())
    for m in metrics:
        if m == 'accuracy':
            stat = accuracy_score(labels_all, pred_all)
        else:
            raise NotImplementedError
        outputs[m] = stat
    return outputs


def get_lr(step, total_steps, warmup_steps, use_cos_decay):
    """
    Util function to get learning rate
    """
    if step < warmup_steps:
        # Linear warmup
        mul = (step + 1) / warmup_steps
        return mul
    else:
        # Cosine decay
        if use_cos_decay:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return (1 + math.cos(math.pi * progress)) / 2
        else:
            return 1


def train_epochs(model, cfg, train_loader, val_loader, writer=None, checkpoint_dir=None):
    """
    General purpose training loops for training a model
    For the purpose of cifar, probably a classifier that is not a noisy classifier
    (noisy classifier training is handled in base_diffusion_manager)

    Note:
        1) model should implement a function `loss` 
        2) Currently track training using tensorboard (also have code to adapt for wandb)
        Can also include customized metrics for evaluation

    """
    if writer is None:
        writer = bookkeeping.DummyWriter('none')
    torch.autograd.set_detect_anomaly(True)
    
    # Number of epochs is either specified directly
    # or is the total number of iterations divided by number of batches
    if 'n_epochs' in cfg.training:
        num_epochs = cfg.training.n_epochs
    else:
        if 'n_iters' not in cfg.training:
            raise ValueError('Either n_epochs or n_iters should be specified for in config.training')
        num_epochs = cfg.training.n_iters // len(train_loader)

    # Define optimizer and schedulers
    lr = cfg.training.lr
    clip_grad = cfg.training.clip_grad
    warmup_steps = cfg.training.warmup
    use_cos_decay = cfg.training.use_cos_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr(step, total_steps, warmup_steps, use_cos_decay)
    )
    loss_fn = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)

    # Define current state and metrics to track
    state = {
        'train_losses': [],
        'val_outputs': dict(),
        'best_epoch': 0
    }
    # Metrics used for storing ckpt
    # If not specify, save the best ckpt according to val loss
    if 'ckpt_metrics' in cfg.training:
        ckpt_metrics = cfg.training.ckpt_metrics
    else:
        ckpt_metrics = ['loss']
    for metric in ckpt_metrics:
        if metric == 'loss':
            state['best_val_loss'] = np.inf
        elif metric in ['accuracy']:
            state[f'best_val_{metric}'] = -np.inf
        else:
            raise ValueError(f'ckpt_metric={metric} not supported')

    # Loop over the training epochs
    # Number of training steps is aggregated over all epochs and minibatchs
    training_state = dict(n_iter=0)

    for epoch in range(num_epochs):
        batch_train_output, training_state = train_single_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, 
            training_state, writer, clip_grad
        )
        # Track average over all the batches
        for k, v in batch_train_output.items():
            avg = np.mean(np.array(v))
            writer.add_scalar(f'{k}/train/avg', avg, epoch)

        # Determine the loss on the validation set for the epoch if the 
        # validation dataloader has been defined (i.e. is not None)
        if val_loader is not None:
            batch_val_output = eval_on_data(
                model, val_loader, loss_fn, cfg.training.val_metrics
            )
            for k, v in batch_val_output.items():
                writer.add_scalar(f'{k}/val', v, epoch)

        # Save checkpoint for model with best metric on val set
        for metric in ckpt_metrics:
            if 'loss' in metric:
                save_ckpt =  batch_val_output[metric] < state[f'best_val_{metric}']
            else:
                save_ckpt =  batch_val_output[metric] > state[f'best_val_{metric}']
            if save_ckpt and checkpoint_dir:
                state[f'best_val_{metric}'] = batch_val_output[metric]
                state['best_epoch'] = epoch
                ckpt_state = {
                    **{
                        'model': model,
                        'optimizer': optimizer,
                        'epoch': epoch,
                        'n_iter': training_state['n_iter'],
                        'train_loss': batch_train_output['loss']
                    },
                    **batch_val_output
                }
                # Save archive checkpoint for model with best validation metric
                bookkeeping.save_archive_checkpoint(
                    checkpoint_dir, ckpt_state, f'ckpt_best_val_{metric}.pt')
                print(f'New best val {metric} at epoch {epoch}: {batch_val_output[metric]:.4f}, saved checkpoint')
