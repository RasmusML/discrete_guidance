import argparse
import os.path
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from concurrent.futures import ProcessPoolExecutor
from applications.inverse_folding.utils.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
from applications.inverse_folding.utils.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
from applications.inverse_folding.utils.model_utils import ProteinFeatures, EncLayer, DecLayer, gather_nodes, cat_neighbors_nodes, FlowMatchPMPNN
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.fm_utils import flow_matching_sampling, flow_matching_sampling_masking_euler, flow_matching_loss_masking, sample_xt
from src.digress_utils import d3pm_loss_masking


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                     S.contiguous().view(-1)).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)  #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument(
        "--path_for_training_data",
        type=str,
        default=
        "/home/hunter/projects/diffusion/ProteinMPNN/training/pdb_2021aug02",
        help="Path to the training data")
    parser.add_argument(
        "--path_for_outputs",
        type=str,
        default=
        "/home/hunter/projects/discrete_diffusion/refactor/inverse_folding_models",
        help="Path to the output directory")
    parser.add_argument("--previous_checkpoint",
                        type=str,
                        default="",
                        help="Path to the previous checkpoint")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=400,
                        help="Number of epochs for training")
    parser.add_argument("--save_model_every_n_epochs",
                        type=int,
                        default=1,
                        help="Save the model every n epochs")
    parser.add_argument("--reload_data_every_n_epochs",
                        type=int,
                        default=4,
                        help="Reload data every n epochs")
    parser.add_argument("--num_examples_per_epoch",
                        type=int,
                        default=200000000,
                        help="Number of examples per epoch")
    parser.add_argument("--batch_size",
                        type=int,
                        default=40000,
                        help="Batch size")
    parser.add_argument("--max_protein_length",
                        type=int,
                        default=2000,
                        help="Maximum protein length")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=128,
                        help="Hidden dimension size")
    parser.add_argument("--num_encoder_layers",
                        type=int,
                        default=4,
                        help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers",
                        type=int,
                        default=4,
                        help="Number of decoder layers")
    parser.add_argument("--num_neighbors",
                        type=int,
                        default=30,
                        help="Number of neighbors")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.1,
                        help="Dropout rate")
    parser.add_argument("--backbone_noise",
                        type=float,
                        default=0.1,
                        help="Backbone noise level")
    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.0,
                        help="label smoothing level")
    parser.add_argument("--rescut",
                        type=float,
                        default=3.5,
                        help="Rescut value")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Enable debug mode")
    parser.add_argument("--gradient_norm",
                        type=float,
                        default=-1.0,
                        help="Gradient norm (-1 for no norm)")
    parser.add_argument("--mixed_precision",
                        type=bool,
                        default=True,
                        help="Use mixed precision")
    parser.add_argument("--timesteps",
                        type=int,
                        default=100,
                        help="Number of DiGress Timesteps")
    parser.add_argument("--use_digress",
                        action='store_true',
                        help="Whether to use digress")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    scaler = torch.amp.GradScaler('cuda')

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if args.use_digress:
        subfolders = ['model_weights_digress']
    else:
        subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST": f"{data_path}/list.csv",
        "VAL": f"{data_path}/valid_clusters.txt",
        "TEST": f"{data_path}/test_clusters.txt",
        "DIR": f"{data_path}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,  #resolution cutoff for PDBs
        "HOMO": 0.70  #min seq.id. to detect homo chains
    }

    LOAD_PARAM = {
        'batch_size': 1,
        'shuffle': True,
        'pin_memory': False,
        'num_workers': 8
    }

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000
    label_smoothing = args.label_smoothing

    train, valid, test = build_training_clusters(params, args.debug)
    print('built clusters')
    print(len(list(train.keys())))
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               worker_init_fn=worker_init_fn,
                                               **LOAD_PARAM)
    #train_loader = torch.utils.data.DataLoader(train_set)
    #train_loader = [b for b in tqdm(train_loader, total=len(train_loader))]
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               worker_init_fn=worker_init_fn,
                                               **LOAD_PARAM)
    #valid_loader = torch.utils.data.DataLoader(valid_set)
    #valid_loader = [b for b in tqdm(valid_loader, total=len(valid_loader))]
    model = FlowMatchPMPNN(node_features=args.hidden_dim,
                           edge_features=args.hidden_dim,
                           hidden_dim=args.hidden_dim,
                           num_encoder_layers=args.num_encoder_layers,
                           num_decoder_layers=args.num_encoder_layers,
                           k_neighbors=args.num_neighbors,
                           dropout=args.dropout,
                           augment_eps=args.backbone_noise,
                           num_letters=21,
                           vocab=21)
    model.to(device)

    if PATH:
        checkpoint = torch.load(PATH, weights_only=False)
        total_step = checkpoint['step']  #write total_step from the checkpoint
        epoch = checkpoint['epoch']  #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    #optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3,
                                  weight_decay=0.0)
    if PATH:
        #optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print('getting pdbs')
    with ProcessPoolExecutor(max_workers=48) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(
                executor.submit(get_pdbs, train_loader, 1,
                                args.max_protein_length,
                                args.num_examples_per_epoch, i == 0))
            p.put_nowait(
                executor.submit(get_pdbs, valid_loader, 1,
                                args.max_protein_length,
                                args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()

        dataset_train = StructureDataset(pdb_dict_train,
                                         truncate=None,
                                         max_length=args.max_protein_length)
        dataset_valid = StructureDataset(pdb_dict_valid,
                                         truncate=None,
                                         max_length=args.max_protein_length)

        loader_train = StructureLoader(dataset_train,
                                       batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid,
                                       batch_size=args.batch_size)

        reload_c = 0
        ################
        ################
        # Training loop

        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(
                        pdb_dict_train,
                        truncate=None,
                        max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train,
                                                   batch_size=args.batch_size)
                    q.put_nowait(
                        executor.submit(get_pdbs, train_loader, 1,
                                        args.max_protein_length,
                                        args.num_examples_per_epoch))
                reload_c += 1
            for _, batch in tqdm(enumerate(loader_train),
                                 total=len(loader_train)):
                start_batch = time.time()
                # in flow matching, T=1 is un-noised data, call sequences S1
                X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                    batch, device)

                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask * chain_M

                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        denoising_model = lambda xt, t: model(
                            X, xt, mask, chain_M, residue_idx,
                            chain_encoding_all)[1]
                        if args.use_digress:
                            loss = d3pm_loss_masking(
                                denoising_model,
                                S1,
                                mask_idx=20,
                                loss_mask=mask_for_loss.to(bool),
                                label_smoothing=label_smoothing,
                                timesteps=args.timesteps,
                                reduction='mean')
                        else:
                            loss = flow_matching_loss_masking(
                                denoising_model,
                                S1,
                                mask_idx=20,
                                loss_mask=mask_for_loss.to(bool),
                                label_smoothing=label_smoothing,
                                reduction='mean')
                        loss_scaled = loss  #* mask_for_loss.sum() / 2000

                    scaler.scale(loss_scaled).backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    denoising_model = lambda xt, t: model(
                        X, xt, mask, chain_M, residue_idx, chain_encoding_all)[
                            1]
                    if args.use_digress:
                        loss = d3pm_loss_masking(
                            denoising_model,
                            S1,
                            mask_idx=20,
                            loss_mask=mask_for_loss.to(bool),
                            label_smoothing=label_smoothing,
                            timesteps=args.timesteps,
                            reduction='mean')
                    else:
                        loss = flow_matching_loss_masking(
                            denoising_model,
                            S1,
                            mask_idx=20,
                            loss_mask=mask_for_loss.to(bool),
                            label_smoothing=label_smoothing,
                            reduction='mean')
                    loss_scaled = loss  #* mask_for_loss.sum() / 2000
                    loss_scaled.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm)

                    optimizer.step()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for ve in range(1):
                    for _, batch in enumerate(loader_valid):
                        X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                            batch, device)

                        B, D = S1.shape
                        t = torch.rand((B, )).to(device)
                        St = S1.clone()
                        St[torch.rand((B, D)).to(device) < (
                            1 - t[:, None]
                        )] = 20  # Corrupt with masks, 22 represents MASK

                        log_probs, logits = model(X, St, mask, chain_M,
                                                  residue_idx,
                                                  chain_encoding_all)
                        mask_for_loss = mask * chain_M
                        loss, loss_av, true_false = loss_nll(
                            S1, log_probs, mask_for_loss)

                        validation_sum += torch.sum(
                            loss * mask_for_loss).cpu().data.numpy()
                        validation_acc += torch.sum(
                            (true_false.to(bool) | (St == S1)).to(float) *
                            mask_for_loss).cpu().data.numpy()
                        validation_weights += torch.sum(
                            mask_for_loss).cpu().data.numpy()

            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            validation_perplexity_ = np.format_float_positional(
                np.float32(validation_perplexity), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3)

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1 - t0),
                                            unique=False,
                                            precision=1)
            with open(logfile, 'a') as f:
                f.write(
                    f'epoch: {e+1}, step: {total_step}, time: {dt}, valid: {validation_perplexity_}, valid_acc: {validation_accuracy_}\n'
                )
            print(
                f'epoch: {e+1}, step: {total_step}, time: {dt}, valid: {validation_perplexity_}, valid_acc: {validation_accuracy_}'
            )

            if args.use_digress:
                checkpoint_filename_last = base_folder + 'model_weights_digress/epoch_last.pt'.format(
                e + 1, total_step)
            else:
                checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'.format(
                e + 1, total_step)
            torch.save(
                {
                    'epoch': e + 1,
                    'step': total_step,
                    'num_edges': args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_filename_last)

            if (e + 1) % args.save_model_every_n_epochs == 0:
                if args.use_digress:
                    checkpoint_filename = base_folder + 'model_weights_digress/epoch{}_step{}.pt'.format(
                    e + 1, total_step)
                else:
                    checkpoint_filename = base_folder + 'model_weights/epoch{}_step{}.pt'.format(
                        e + 1, total_step)
                torch.save(
                    {
                        'epoch': e + 1,
                        'step': total_step,
                        'num_edges': args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_filename)
