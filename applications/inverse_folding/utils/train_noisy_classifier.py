import torch
import torch.nn as nn
import torch.nn.functional as F
from applications.inverse_folding.utils.model_utils import FlowMatchPMPNN, StabilityPMPNN
from applications.inverse_folding.utils.data import aa_to_i, i_to_aa, seq_to_one_hot, RocklinDataset, process_rocklin_data, generate_train_val_split, rocklin_df_to_dataset, ROCKLIN_DIR, ProteinGymDataset
from training.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
import pandas as pd
import numpy as np
from training.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam, AdamW
import scipy.stats
import argparse
from src.digress_utils import d3pm_sample_xt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=Path, default='./pretrained_weights/stability_regression.pt')
    parser.add_argument('--cluster', type=str, default='146')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--use_digress', action='store_true')
    parser.add_argument('--timesteps', type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda'
    name = args.cluster
    threshold = args.threshold

    rocklin_df = process_rocklin_data()
    noisy_classifier = StabilityPMPNN.init()
    fn = args.input_path
    noisy_classifier.load_state_dict(torch.load(fn, weights_only=False))
    noisy_classifier.to(device)

    train_lr = 1e-4
    weight_decay = 0.0
    adam_betas = (0.9, 0.99)
    opt = AdamW(noisy_classifier.parameters(),
                lr=train_lr,
                betas=adam_betas,
                weight_decay=weight_decay)

    criterion = torch.nn.BCEWithLogitsLoss()

    single_df = rocklin_df[rocklin_df.WT_cluster == name]
    single_dataset = rocklin_df_to_dataset(single_df)

    loader_train = StructureLoader(single_dataset, batch_size=24000)

    accuracy = 0.0
    accuracy_clean = 0.0
    for epoch in range(args.nepoch):
        noisy_classifier.train()
        train_pbar = tqdm(loader_train)
        #train_pbar = (loader_train)
        losses_per_batch = np.zeros(len(loader_train))
        train_ys = []
        train_yhats = []
        train_yhats_clean = []
        for batch_idx, batch in enumerate(train_pbar):
            opt.zero_grad()
            X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                batch, device)
            B, D = S1.shape
            if args.use_digress:
                t = torch.randint(low=1, high=args.timesteps + 1, size=(B, )).float().to(S1.device)
                St = d3pm_sample_xt(S1, t, 20, args.timesteps, pad_idx=None)
            else:
                # sample times
                t = torch.rand((B, )).to(device)
                St = S1.clone()
                sampled_mask = torch.rand((B, D)).to(device) < (1 - t[:, None])
                St[sampled_mask] = 20  # Corrupt with masks, 21 represents MASK
            for b in batch:
                b['seq'] = b['wt']
                b['seq_chain_A'] = b['wt']
            #X, S1_wt, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            y = torch.tensor([b['y'] for b in batch]).to(device)
            wt_y = torch.tensor([b['wt_dG'] for b in batch]).to(device)
            yhat = noisy_classifier(X, St, mask, chain_M, residue_idx,
                                    chain_encoding_all).reshape(-1)
            #yhat_wt = noisy_classifier(X, S1_wt, mask, chain_M, residue_idx,chain_encoding_all).reshape(-1)
            with torch.no_grad():
                yhat_clean = noisy_classifier(X, S1, mask, chain_M,
                                              residue_idx,
                                              chain_encoding_all).reshape(-1)
                yhat_clean_logit = yhat_clean
                yhat_clean_label = (F.sigmoid(yhat_clean_logit) > 0.5).to(
                    torch.float)
            ylabel = ((y - wt_y) >= threshold).to(torch.float)
            #yhat_logit = (yhat - yhat_wt)
            yhat_logit = yhat
            yhat_label = (F.sigmoid(yhat_logit) > 0.5).to(torch.float)
            loss = criterion(yhat_logit, ylabel)
            #loss += criterion(yhat_clean_logit, ylabel)
            loss.backward()
            opt.step()
            losses_per_batch[batch_idx] = loss.item()
            train_pbar.set_postfix(loss=losses_per_batch[:batch_idx].mean(),
                                   accuracy=accuracy,
                                   accuracy_clean=accuracy_clean)
            train_ys.append(ylabel.cpu().detach().numpy())
            train_yhats.append(yhat_label.cpu().detach().numpy())
            train_yhats_clean.append(yhat_clean_label.cpu().detach().numpy())
            accuracy = (np.concatenate(train_ys) == np.concatenate(train_yhats)
                        ).mean()
            accuracy_clean = (np.concatenate(train_ys) == np.concatenate(
                train_yhats_clean)).mean()
    if args.use_digress:
        torch.save(noisy_classifier.state_dict(), f'noisy_classifier_digress_{name}_{args.nepoch}.pt')
    else:
        torch.save(noisy_classifier.state_dict(), f'noisy_classifier_{name}_{args.nepoch}.pt')
