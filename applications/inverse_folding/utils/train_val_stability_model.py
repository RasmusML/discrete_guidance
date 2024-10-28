from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from applications.inverse_folding.utils.model_utils import FlowMatchPMPNN, StabilityPMPNN
from applications.inverse_folding.utils.data import aa_to_i, i_to_aa, seq_to_one_hot, RocklinDataset, process_rocklin_data, generate_train_val_split, rocklin_df_to_dataset, ROCKLIN_DIR, ProteinGymDataset
from applications.inverse_folding.utils.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
import pandas as pd
import numpy as np
from applications.inverse_folding.utils.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
from tqdm import tqdm

from torch.optim import Adam, AdamW
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=
        "/home/hunter/projects/discrete_diffusion/refactor/inverse_folding_models/model_weights/epoch_last.pt")
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
    return parser.parse_args()



args = parse_args()
device = 'cuda'
rocklin_df = process_rocklin_data()
name_to_graph = torch.load(ROCKLIN_DIR / 'name_to_graph.pt')

# Save loaders once since they take a long time to compute
if True:
    batch_size = 6000*2
    train_ds, val_ds, train_names, val_names = generate_train_val_split(
        rocklin_df)
    loader_train = StructureLoader(train_ds, batch_size=batch_size)

    loader_val = StructureLoader(val_ds, batch_size=batch_size)

    torch.save(
        loader_train, ROCKLIN_DIR / 'train_loader.pt'
    )
    torch.save(
        loader_val, ROCKLIN_DIR / 'val_loader.pt'
    )

# load loaders
loader_train = torch.load(
    ROCKLIN_DIR / 'train_loader.pt'
)
loader_val = torch.load(
    ROCKLIN_DIR / 'val_loader.pt'
)

val_names = list(set([x['cluster'] for b in tqdm(loader_val) for x in b]))
train_names = list(set([x['cluster'] for b in tqdm(loader_train) for x in b]))

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

stability_model = StabilityPMPNN(model)
stability_model.load_fm_mpnn_weights(
    args.model_path
)
stability_model.to(device)

train_lr = 1e-4
weight_decay = 1e-4
adam_betas = (0.9, 0.99)
opt = AdamW(stability_model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=weight_decay)

criterion = torch.nn.MSELoss()
batch_size = 6000 * 2
val_spearman = 0
val_spearman_epoch = []
for epoch in range(10):
    stability_model.train()
    train_pbar = tqdm(loader_train)
    #train_pbar = (loader_train)
    losses_per_batch = np.zeros(len(loader_train))
    train_ys = []
    train_yhats = []

    for batch_idx, batch in enumerate(train_pbar):
        opt.zero_grad()
        X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
            batch, device)
        for b in batch:
            b['seq'] = b['wt']
            b['seq_chain_A'] = b['wt']
        X, S1_wt, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
            batch, device)
        y = torch.tensor([b['y'] for b in batch]).to(device)
        wt_y = torch.tensor([b['wt_dG'] for b in batch]).to(device)
        yhat = stability_model(X, S1, mask, chain_M, residue_idx,
                               chain_encoding_all).reshape(-1)
        yhat_wt = stability_model(X, S1_wt, mask, chain_M, residue_idx,
                                  chain_encoding_all).reshape(-1)
        loss = criterion(yhat - yhat_wt, y - wt_y)
        #loss = criterion(yhat, batch.y)
        loss.backward()
        opt.step()
        losses_per_batch[batch_idx] = loss.item()
        #train_pbar.set_postfix(loss=losses_per_batch[:batch_idx].mean(), val_spearman=val_spearman)
        train_ys.append(y.cpu().detach().numpy())
        train_yhats.append(yhat.cpu().detach().numpy())

    stability_model.eval()
    val_spearmans_by_group = []
    for name in val_names:
        ds_by_cluster = rocklin_df_to_dataset(
            rocklin_df[rocklin_df.WT_cluster == name])
        val_dl = StructureLoader(ds_by_cluster, batch_size=6000 * 2)

        ys = []
        yhats = []
        val_pbar = val_dl
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                    batch, device)
                yhat = stability_model(X, S1, mask, chain_M, residue_idx,
                                       chain_encoding_all).reshape(-1)
                y = torch.tensor([b['y'] for b in batch])
                ys.append(y.cpu().numpy())
                yhats.append(yhat.cpu().numpy())
        ys = np.concatenate(ys)
        yhats = np.concatenate(yhats)
        val_spearman = scipy.stats.spearmanr(yhats, ys).statistic
        val_spearmans_by_group.append(val_spearman)

    val_spearman = np.mean(val_spearmans_by_group)
    print(f"{epoch}: {val_spearman}")
    val_spearman_epoch.append(val_spearman)
    stability_model.train()
    torch.save(stability_model.state_dict(),
               f'stability_train_val_checkpoint_{epoch}.pt')
