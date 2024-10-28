import torch
import copy
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np

ROCKLIN_DIR = Path(__file__).parent / 'rocklin_data'

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
i_to_aa = {i: aa for i, aa in enumerate(alphabet)}
aa_to_i = {aa: i for i, aa in enumerate(alphabet)}


def seq_to_one_hot(s):
    return F.one_hot(torch.tensor([aa_to_i[aa] for aa in s]), num_classes=20)


class RocklinDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(
        self,
        tups,
    ):
        super().__init__()
        'Initialization'
        self.tups = tups

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return len(self.tups)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        aa_seq, graph, dG, wt_dG = self.tups[index]
        graph = copy.deepcopy(graph)
        graph['wt'] = graph['seq']
        graph['seq'] = aa_seq
        graph['masked_list'] = ['A']
        graph['visible_list'] = []
        graph['seq_chain_A'] = aa_seq
        graph['x'] = seq_to_one_hot(aa_seq).float()
        graph['y'] = torch.tensor(dG)
        graph['wt_dG'] = torch.tensor(wt_dG)
        return graph


class ProteinGymDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, graph, seqs):
        super().__init__('.', None, None, None)
        'Initialization'
        self.graph = graph
        self.seqs = seqs

    def len(self):
        'Denotes the total number of samples'
        return len(self.seqs)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        seq = self.seqs[index]
        graph = self.graph.clone()
        graph.wt = graph.x.clone()
        graph.x = seq_to_one_hot(seq).float()
        graph.extra_x[:, 0] = 0.0
        return graph


def process_rocklin_data():
    if False:
        csv_fn = ROCKLIN_DIR / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
        rocklin_df = pd.read_csv(csv_fn)
        rocklin_df = rocklin_df[rocklin_df.dG_ML !=
                                '-']  # filter unreliable delta Gs
        rocklin_df.loc[rocklin_df.dG_ML == '>5', "dG_ML"] = '5'
        rocklin_df.loc[rocklin_df.dG_ML == '<-1', "dG_ML"] = '-1'
        rocklin_df.dG_ML = rocklin_df.dG_ML.astype(float)
        # remove insertions and deletions
        rocklin_df = rocklin_df[rocklin_df.mut_type.map(
            lambda x: ('ins' not in x) and ('del' not in x))]
        rocklin_df = rocklin_df.loc[:, [
            'name', 'aa_seq', 'mut_type', 'WT_cluster', 'WT_name', 'dG_ML',
            'ddG_ML', 'Stabilizing_mut', 'pair_name'
        ]]
        aa_to_dG = {
            aa: dG
            for aa, dG in rocklin_df.groupby(
                'aa_seq').dG_ML.mean().reset_index().values
        }
        rocklin_df['avg_dG'] = rocklin_df.aa_seq.map(lambda aa: aa_to_dG[aa])
        aa_seq_to_remove = rocklin_df[(
            rocklin_df.avg_dG - rocklin_df.dG_ML).abs() > 0.5].aa_seq.values
        # remove sequences with high discrepancy between replicates
        rocklin_df = rocklin_df[rocklin_df.aa_seq.map(
            lambda x: x not in aa_seq_to_remove)]
        # drop duplicates
        rocklin_df = rocklin_df.loc[rocklin_df.aa_seq.drop_duplicates().index]
        rocklin_df.WT_cluster = rocklin_df.WT_cluster.astype(str)
    else:
        csv_fn = ROCKLIN_DIR / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
        rocklin_df = pd.read_csv(csv_fn, low_memory=False)
        rocklin_df = rocklin_df[rocklin_df.dG_ML !=
                                '-']  # filter unreliable delta Gs
        rocklin_df.loc[rocklin_df.dG_ML == '>5', "dG_ML"] = '5'
        rocklin_df.loc[rocklin_df.dG_ML == '<-1', "dG_ML"] = '-1'
        rocklin_df.dG_ML = rocklin_df.dG_ML.astype(float)
        # remove insertions and deletions
        rocklin_df = rocklin_df[rocklin_df.mut_type.map(
            lambda x: ('ins' not in x) and ('del' not in x))]
        rocklin_df = rocklin_df.loc[:, [
            'name', 'aa_seq', 'mut_type', 'WT_cluster', 'WT_name', 'dG_ML',
            'ddG_ML', 'Stabilizing_mut', 'pair_name'
        ]]
        aa_to_dG = {
            aa: dG
            for aa, dG in rocklin_df.groupby(
                'aa_seq').dG_ML.mean().reset_index().values
        }
        rocklin_df['avg_dG'] = rocklin_df.aa_seq.map(lambda aa: aa_to_dG[aa])
        rocklin_df = rocklin_df.loc[
            rocklin_df.loc[:, ['aa_seq', 'WT_name']].drop_duplicates().index]
        aa_seq_to_remove = set(
            rocklin_df[(rocklin_df.avg_dG -
                        rocklin_df.dG_ML).abs() > 1.0].aa_seq.values)
        rocklin_df = rocklin_df[rocklin_df.aa_seq.map(
            lambda x: x not in aa_seq_to_remove)]
        rocklin_df.WT_cluster = rocklin_df.WT_cluster.astype(str)
        wt_name_to_dG = {
            n: dG
            for n, dG in rocklin_df[rocklin_df.mut_type ==
                                    'wt'].loc[:, ['WT_name', 'avg_dG']].values
        }
        rocklin_df['WT_dG'] = rocklin_df.WT_name.map(wt_name_to_dG)
        rocklin_df = rocklin_df[
            ~rocklin_df.WT_dG.isna()]  # should throw out ~400 sequcnes
    return rocklin_df


def generate_train_val_split(rocklin_df):
    name_to_graph = torch.load(ROCKLIN_DIR / 'name_to_graph.pt', weights_only=False)
    names, counts = zip(
        *rocklin_df.WT_cluster.value_counts().reset_index().values
    )  #zip(*rocklin_df.WT_cluster.map(str).value_counts().reset_index().values)
    names = [str(n) for n in names]
    counts = np.asarray(counts)
    probs = counts / counts.sum()
    name_to_probs = {n: p for n, p in zip(names, probs)}

    train_names = []
    val_names = []
    while sum([name_to_probs[n] for n in train_names]) < 0.8:
        name = np.random.choice(list(set(names) - set(train_names)))
        if name not in train_names:
            train_names.append(name)
    val_names = list(set(names) - set(train_names))

    train_data = []
    val_data = []
    for row in rocklin_df.loc[:, [
            'name', 'aa_seq', 'WT_cluster', 'WT_name', 'dG_ML', 'WT_dG'
    ]].values:
        name, aa_seq, cluster, pdb, dG, wt_dG = row
        pdb_stem = pdb.replace("|", ':').split(".pdb")[0]
        assert pdb_stem in name_to_graph
        graph = name_to_graph[pdb_stem]
        assert len(aa_seq) == len(graph['seq'])
        assert (cluster in train_names) or (cluster in val_names)
        graph['cluster'] = cluster
        if cluster in train_names:
            train_data.append((aa_seq, graph, float(dG), float(wt_dG)))
        if cluster in val_names:
            val_data.append((aa_seq, graph, float(dG), float(wt_dG)))

    train_ds = RocklinDataset(train_data)
    val_ds = RocklinDataset(val_data)
    return train_ds, val_ds, train_names, val_names


def rocklin_df_to_dataset(rocklin_df):
    name_to_graph = torch.load(ROCKLIN_DIR / 'name_to_graph.pt', weights_only=False)
    data = []
    for row in rocklin_df.loc[:, [
            'name', 'aa_seq', 'WT_cluster', 'WT_name', 'dG_ML', 'WT_dG'
    ]].values:
        name, aa_seq, cluster, pdb, dG, wt_dG = row
        pdb_stem = pdb.replace("|", ':').split(".pdb")[0]
        assert pdb_stem in name_to_graph
        graph = name_to_graph[pdb_stem]
        assert len(aa_seq) == len(graph['seq'])
        data.append((aa_seq, graph, float(dG), float(wt_dG)))
    ds = RocklinDataset(data)
    return ds


def df_to_dataset(df):
    name_to_graph = torch.load(ROCKLIN_DIR / 'name_to_graph.pt', weights_only=False)
    data = []
    for row in df.loc[:, ['seqs', 'labels', 'WT_name']].values:
        aa_seq, label, pdb = row
        pdb_stem = pdb.replace("|", ':').split(".pdb")[0]
        assert pdb_stem in name_to_graph
        graph = name_to_graph[pdb_stem]
        assert len(aa_seq) == len(graph['seq'])
        data.append((aa_seq, graph, label))
    ds = IFDataset(data)
    return ds


class IFDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(
        self,
        tups,
    ):
        super().__init__()
        'Initialization'
        self.tups = tups

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return len(self.tups)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        aa_seq, graph, label = self.tups[index]
        graph = copy.deepcopy(graph)
        graph['wt'] = graph['seq']
        graph['seq'] = aa_seq
        graph['masked_list'] = ['A']
        graph['visible_list'] = []
        graph['seq_chain_A'] = aa_seq
        graph['x'] = seq_to_one_hot(aa_seq).float()
        graph['y'] = torch.tensor(label)
        return graph

