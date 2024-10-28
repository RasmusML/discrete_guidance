import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from applications.inverse_folding.utils.model_utils import FlowMatchPMPNN, StabilityPMPNN, featurize
from applications.inverse_folding.utils.data import aa_to_i, i_to_aa, process_rocklin_data, rocklin_df_to_dataset
from applications.inverse_folding.utils.utils import StructureDataset, StructureLoader
from src.fm_utils import flow_matching_sampling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x1_temp', type=float, default=1.0)
    parser.add_argument('--guide_temp', type=float, default=1.0)
    parser.add_argument('--stochasticity', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cluster', type=str, default='146')
    parser.add_argument('--fm_weights', type=Path, default='./pretrained_weights/epoch_last.pt')
    parser.add_argument('--predictor_weights', type=Path, required=True)
    parser.add_argument('--use_tag', action='store_true')
    parser.add_argument('--do_purity_sampling', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cluster = args.cluster

    x1_temp = args.x1_temp
    guide_temp = args.guide_temp
    stochasticity = args.stochasticity
    dt = args.dt
    num_samples = args.num_samples
    model_path = args.fm_weights
    use_tag = args.use_tag
    do_purity_sampling = args.do_purity_sampling

    rocklin_df = process_rocklin_data()

    device = 'cuda'
    S = 21    
    bs = args.batch_size
    single_df = rocklin_df[rocklin_df.WT_cluster == cluster]
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
    model.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        h_V, h_E, E_idx, mask_attend = model.encode_structure(
            X, mask, chain_M, residue_idx, chain_encoding_all)
        denoising_model = lambda xt, t: model.decode(h_V, h_E, E_idx, xt, mask_attend, mask, chain_M)[1]

    noisy_classifier = StabilityPMPNN.init()
    noisy_classifier.load_state_dict(torch.load(args.predictor_weights))
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


    samples = flow_matching_sampling(num_samples, denoising_model, S=S, D=D, device=device, mask_idx=S-1, stochasticity=stochasticity, batch_size=bs, x1_temp=x1_temp, dt=dt, predictor_log_prob=predictor_log_prob, guide_temp=guide_temp, use_tag=use_tag, do_purity_sampling=do_purity_sampling)
    seqs = [''.join([i_to_aa[i] for i in x]) for x in samples]

    outfn = f'guidance_x1-temp_{x1_temp}_guide-temp_{guide_temp}_use-tag_{use_tag}.fa'
    SeqIO.write([
        SeqRecord(Seq(s), id=f'sample_{i}', description=f'guide-temp_{guide_temp}')
        for i, s in enumerate(seqs)
    ], outfn, 'fasta')
    
