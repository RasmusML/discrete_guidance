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
import pandas as pd
from src.fm_utils import flow_matching_sampling
from applications.inverse_folding.utils.data import df_to_dataset
from src.digress_utils import d3pm_sample_xt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str, default='146')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--use_digress', action='store_true')
    parser.add_argument('--timesteps', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    cluster = args.cluster
    use_digress = args.use_digress
    x1_temp = 1.0
    guide_temp = 0.1
    stochasticity = 0 if use_digress else 1.0
    timesteps = args.timesteps
    dt = 0.02
    num_samples = 100
    model_path = '/home/hunter/projects/discrete_diffusion/refactor/inverse_folding_models/model_weights/epoch_last.pt'
    use_tag = True
    do_purity_sampling = False

    device = 'cuda'
    S = 21    

    bs = 100

    rocklin_df = process_rocklin_data()
    sub_df = rocklin_df[rocklin_df.WT_cluster == cluster]
    wt_name = sub_df['WT_name'].iloc[0]

    single_df = rocklin_df[(rocklin_df.WT_cluster == cluster) & (rocklin_df.WT_name == wt_name)]
    single_dataset = rocklin_df_to_dataset(single_df)
    wt_ds = rocklin_df_to_dataset(
        pd.concat([single_df.iloc[0:1] for _ in range(bs)]))
    D = len(wt_ds[0]['seq_chain_A'])
    wt_dl = StructureLoader(wt_ds, batch_size=D*bs)
    assert (len(wt_dl) == 1)
    wt_batch = next(iter(wt_dl))
    X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
        wt_batch, device)

    model = FlowMatchPMPNN(vocab=21,
                           num_letters=21)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        h_V, h_E, E_idx, mask_attend = model.encode_structure(
            X, mask, chain_M, residue_idx, chain_encoding_all)
        denoising_model = lambda xt, t: model.decode(h_V, h_E, E_idx, xt, mask_attend, mask, chain_M)[1]



    wt_dl = StructureLoader(wt_ds, batch_size=D*bs)
    assert (len(wt_dl) == 1)
    wt_batch = next(iter(wt_dl))
    X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
        wt_batch, device)

    clean_classifier = StabilityPMPNN.init()
    fn = './pretrained_weights/stability_regression.pt'
    clean_classifier.load_state_dict(torch.load(fn))
    clean_classifier.to(device)
    with torch.no_grad():
        h_V_cl, h_E_cl, E_idx_cl, mask_attend_cl = clean_classifier.encode_structure(
            X, mask, chain_M, residue_idx, chain_encoding_all)
        # lambda function + hack to expand the batch dimension to match xt
        clean_log_prob = lambda xt, t: clean_classifier.decode(h_V_cl[0:1].repeat(xt.shape[0], 1, 1), h_E_cl[0:1].repeat(xt.shape[0], 1, 1, 1), E_idx_cl[0:1].repeat(xt.shape[0], 1, 1), xt, mask_attend_cl[0:1].repeat(xt.shape[0], 1, 1), mask[0:1].repeat(xt.shape[0], 1), chain_M[0:1].repeat(xt.shape[0], 1))



    device = 'cuda'
    noisy_classifier = StabilityPMPNN.init()
    if use_digress:
        fn = f'./pretrained_weights/noisy_classifier_digress_{args.cluster}_30.pt'
    else:
        fn = f'./pretrained_weights/noisy_classifier_{args.cluster}_30.pt'
    noisy_classifier.load_state_dict(torch.load(fn))

    if use_tag:
        # need one-hot-encoded input to take gradients
        layer = nn.Linear(21, 128, bias=False)
        layer.weight.data = noisy_classifier.fm_mpnn.W_s.weight.data.T.clone()
        noisy_classifier.fm_mpnn.W_s = layer
    noisy_classifier.to(device)
    with torch.no_grad():
        h_V_c, h_E_c, E_idx_c, mask_attend_c = noisy_classifier.encode_structure(
            X, mask, chain_M, residue_idx, chain_encoding_all)
        # lambda function + hack to expand the batch dimension to match xt
        predictor_log_prob = lambda xt, t: torch.log(F.sigmoid(noisy_classifier.decode(h_V_c[0:1].repeat(xt.shape[0], 1, 1), h_E_c[0:1].repeat(xt.shape[0], 1, 1, 1), E_idx_c[0:1].repeat(xt.shape[0], 1, 1), xt, mask_attend_c[0:1].repeat(xt.shape[0], 1, 1), mask[0:1].repeat(xt.shape[0], 1), chain_M[0:1].repeat(xt.shape[0], 1))))

    num_samples = 1000
    dt = 0.02
    guide_temp = 0.1
    samples = flow_matching_sampling(num_samples, denoising_model, S=S, D=D, device=device, mask_idx=S-1, stochasticity=stochasticity, batch_size=bs, x1_temp=x1_temp, dt=dt, predictor_log_prob=predictor_log_prob, guide_temp=guide_temp, use_tag=use_tag, do_purity_sampling=do_purity_sampling)
    samples_as_seqs = [''.join([i_to_aa[i] for i in x]) for x in samples]        


    wt_dl = StructureLoader(wt_ds, batch_size=D*bs)
    assert (len(wt_dl) == 1)
    wt_batch = next(iter(wt_dl))
    X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
        wt_batch, device)
    #samples = samples[:1000]
    with torch.no_grad():
        if use_tag:
            yhat = torch.exp(predictor_log_prob(F.one_hot(torch.tensor(samples).long().cuda(), num_classes=S).float(), None))
        else:
            yhat = torch.exp(predictor_log_prob(torch.tensor(samples).long().cuda(), None))
    with torch.no_grad():
        ytrue = (clean_log_prob(torch.tensor(samples).long().cuda(), None) - clean_log_prob(S1.repeat(len(samples) // bs, 1).cuda(), None))
    print((ytrue > 0).sum().item(), (yhat > 0.5).sum().item())

    seqs_exp = list(single_df.aa_seq.values)
    seqs_exp_enc = torch.tensor([[aa_to_i[aa] for aa in s] for s in seqs_exp])
    with torch.no_grad():
        ytrue_exp = (clean_log_prob(torch.tensor(seqs_exp_enc).long().cuda(), None) - clean_log_prob(S1[0].repeat(len(seqs_exp_enc), 1).cuda(), None))
        labels_exp = (ytrue_exp >= 0).int().reshape(-1).cpu().numpy()

    with torch.no_grad():
        ytrue_sam = (clean_log_prob(torch.tensor(samples).long().cuda(), None) - clean_log_prob(S1[0].repeat(len(samples), 1).cuda(), None))
        labels_sam = (ytrue_sam >= 0).int().reshape(-1).cpu().numpy()        


    wt_name = single_df['WT_name'].iloc[0]
    aug_df = pd.DataFrame(zip((seqs_exp + samples_as_seqs), np.concatenate((labels_exp, labels_sam))), columns=['seqs', 'labels'])
    aug_df['WT_name'] = wt_name
    train_ds = df_to_dataset(aug_df)
    loader_train = StructureLoader(train_ds, batch_size=24000)        

    noisy_classifier = StabilityPMPNN.init()
    fn = './pretrained_weights/stability_regression.pt'
    noisy_classifier.load_state_dict(torch.load(fn))
    noisy_classifier.to(device)


    train_lr = 1e-4
    weight_decay = 0.0
    adam_betas = (0.9, 0.99)
    opt = AdamW(noisy_classifier.parameters(),
                lr=train_lr,
                betas=adam_betas,
                weight_decay=weight_decay)

    criterion = torch.nn.BCEWithLogitsLoss()

    accuracy = 0.0
    accuracy_clean = 0.0
    for epoch in range(30):
        noisy_classifier.train()
        train_pbar = tqdm(loader_train)
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
                t = torch.rand((B, )).to(device)
                St = S1.clone()
                sampled_mask = torch.rand((B, D)).to(device) < (1 - t[:, None])
                St[sampled_mask] = 20  # Corrupt with masks, 21 represents MASK

            ytrue = torch.tensor([b['y'] for b in batch], device=device).float()
            yhat = noisy_classifier(X, St, mask, chain_M, residue_idx,
                                    chain_encoding_all).reshape(-1)
            loss = criterion(yhat, ytrue)
            loss.backward()
            opt.step()
    noisy_classifier.eval()

    if args.use_digress:
        torch.save(noisy_classifier.state_dict(), f'noisy_classifier_digress_{cluster}_iter_1.pt')    
    else:
        torch.save(noisy_classifier.state_dict(), f'noisy_classifier_{cluster}_iter_1.pt')

