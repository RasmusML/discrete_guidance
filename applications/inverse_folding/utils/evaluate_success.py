import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from Bio import SeqIO
import biotite.structure as struc
import biotite.structure.io as strucio
from applications.inverse_folding.utils.model_utils import (
    FlowMatchPMPNN,
    StabilityPMPNN,
    featurize,
)
from applications.inverse_folding.utils.data import (
    aa_to_i,
    i_to_aa,
    seq_to_one_hot,
    RocklinDataset,
    process_rocklin_data,
    generate_train_val_split,
    rocklin_df_to_dataset,
    ROCKLIN_DIR,
    ProteinGymDataset,
)
from applications.inverse_folding.utils.structure import (
    StructureDataset,
    StructureLoader,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--input_fasta", type=Path, required=True)
    parser.add_argument("--input_colabfold", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument(
        "--stability_weights",
        type=Path,
        default=Path(__file__).parent.parent
        / "pretrained_weights"
        / "stability_regression.pt",
    )
    args = parser.parse_args()
    return args


def dist_to_wt(x, y):
    return sum([aa1 != aa2 for aa1, aa2 in zip(x, y)])


def pairwise_hamming(seqs):
    dists = []
    for i in range(len(seqs) - 1):
        for j in range(i + 1, len(seqs)):
            s1 = seqs[i]
            s2 = seqs[j]

            dist = dist_to_wt(s1, s2)
            dists.append(dist)
    dists = np.asarray(dists)
    return dists.mean()


def get_rmsd(ref, fn):
    query = strucio.load_structure(fn)
    query = query[struc.filter_backbone(query)]
    superimposed, _ = struc.superimpose(ref, query)
    rmsd = struc.rmsd(ref, superimposed)
    return rmsd


def get_wt_pdb(single_df):
    pdb_fn = single_df.iloc[0]["name"]
    pdb_fn = pdb_fn.replace("|", ":")
    pdb_fn = Path(__file__).parent / "rocklin_data" / "AlphaFold_model_PDBs" / pdb_fn
    assert pdb_fn.exists()
    ref = strucio.load_structure(pdb_fn)
    ref = ref[struc.filter_backbone(ref)]
    return ref


if __name__ == "__main__":
    args = parse_args()
    stability_weights_fn = args.stability_weights
    assert stability_weights_fn.exists()

    rocklin_df = process_rocklin_data()
    single_df = rocklin_df[rocklin_df.WT_cluster == args.cluster]

    device = "cuda"
    stability_model = StabilityPMPNN.init(num_encoder_layers=4, num_decoder_layers=4)
    stability_model.load_state_dict(
        torch.load(args.stability_weights, weights_only=False)
    )
    stability_model.to(device)
    stability_model.eval()

    seqs = list(SeqIO.parse(args.input_fasta, "fasta"))

    # Evaluate the stability of the WT sequence
    wt_ds = rocklin_df_to_dataset(
        pd.concat([single_df.iloc[0:1] for _ in range(len(seqs))])
    )
    wt_dl = StructureLoader(wt_ds, batch_size=12000)
    assert len(wt_dl) == 1
    batch = next(iter(wt_dl))
    X, S1, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = (
        featurize(batch, device)
    )
    B, D = S1.shape
    with torch.no_grad():
        wt_yhat = stability_model(
            X, S1, mask, chain_M, residue_idx, chain_encoding_all
        ).reshape(-1)
        print(wt_yhat[0].item())

    # Evaluate the stability of the generated sequences
    names = [s.id for s in seqs]
    seqs = [str(s.seq) for s in seqs]
    with torch.no_grad():
        S1 = torch.tensor([[aa_to_i[aa] for aa in s] for s in seqs]).cuda()
        yhat = stability_model(
            X, S1, mask, chain_M, residue_idx, chain_encoding_all
        ).reshape(-1)

        ddG_hat = yhat - wt_yhat[0].item()
        stability_success = (yhat > wt_yhat[0].item()).cpu().numpy().astype(int)
        diversity = pairwise_hamming(seqs)

    ref = get_wt_pdb(single_df)
    rmsds = []
    for fn in sorted(
        list((args.input_colabfold).glob("sample*_relaxed_rank_001_*.pdb")),
        key=lambda x: int(x.name.split("sample_")[1].split("_guide-temp")[0]),
    ):
        rmsd = get_rmsd(ref, fn)
        rmsds.append(rmsd)

    rmsds = np.asarray(rmsds)
    rmsd_success = (rmsds < 2.0).astype(int)
    df = pd.DataFrame(
        {
            "name": names,
            "seqs": seqs,
            "stability_success": stability_success,
            "rmsd_success": rmsd_success,
        }
    )
    df["success"] = df["stability_success"] * df["rmsd_success"]
    df.to_csv(args.output_csv, index=False)

    print(f"Stability Success: {int(df['stability_success'].sum())} / {len(df)}")
    print(f"RMSD Success: {int(df['rmsd_success'].sum())} / {len(df)}")
    print(f"Success: {int(df['success'].sum())} / {len(df)}")
    print(f"Diversity: {diversity}")
