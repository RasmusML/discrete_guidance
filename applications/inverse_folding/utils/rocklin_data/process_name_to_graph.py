from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from protein_mpnn_utils import parse_PDB

name_to_graph = {}
for pdb_fn in tqdm(list(Path('./AlphaFold_model_PDBs/').glob("*.pdb"))):
    name = pdb_fn.stem
    assert(name not in name_to_graph)
    graph = parse_PDB(pdb_fn.as_posix())[0]
    name_to_graph[name] = graph
torch.save(name_to_graph, 'name_to_graph.pt')
