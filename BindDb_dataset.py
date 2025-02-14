import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset,DataLoader
from pathlib import Path
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from utils import to_network, truncted_BFS, path2mp, mol_paths
import torch.nn.functional as F
from torch_scatter import scatter


max_seq_len = 1000
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict[':'] = 27
seq_dict_len = len(seq_dict)
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        if ch == ":":
            continue
        x[i] = seq_dict[ch]
    return x

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

# zinc_bond_dict = {'NONE':0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}
bonds_dec = {0: '-', 1: '=', 2: '#', 3: '~'}
qm9_node_type={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,'B': 5, 'Si': 6, 'Ru':7,'As':8, 'Pt' : 9,'Fe':10,'Se':11}
qm9_id2ele={v:k for k,v in qm9_node_type.items()}


with open('metapaths.txt', 'r') as fin:
    mp = fin.read().split('\n')
    mp = [i.strip('\'').strip(' \'') for i in mp]
    MP_corpus = set(mp)
MP2id = {i: k for k, i in enumerate(MP_corpus)}



CHAR_SMI_SET = {'[': 0, '%': 1, '2': 2, '(': 3, '=': 4, 'F': 5, 'c': 6, ']': 7, 'r': 8,
                'u': 9, 'M': 10, 'b': 11, '5': 12, 'l': 13, '0': 14, 'dummy': 15, 'I': 16,
                ')': 17, 'N': 18, '~': 19, 't': 20, '6': 21, '\\': 22, '9': 23, '4': 24,
                'O': 25, '#': 26, '8': 27, '7': 28, 'C': 29, 'S': 30, '.': 31, '3': 32,
                'o': 33, 'R': 34, 'H': 35, 'h': 36, '1': 37, '-': 38, 'T': 39, '+': 40,
                '/': 41, 'V': 42, 'Z': 43, 's': 44, 'i': 45, 'g': 46, 'e': 47, 'n': 48,
                'A': 49, 'B': 50, '@': 51, 'P': 52}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch]

    return X


max_seq_len = 1000
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict[':'] = 27
seq_dict_len = len(seq_dict)
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        if ch == ":":
            continue
        x[i] = seq_dict[ch]
    return x

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_path,
                 phase,
                 transform=None,
                 pre_transform=None,
                 ):
        self.data_path = Path(data_path)
        self.phase = phase

        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.transform, self.pre_transform = transform, pre_transform


    def get(self, idx):
        mydata = torch.load(os.path.join(self.processed_dir, f'{self.phase}.pt'))
        return mydata

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.phase}.pt']

    def download(self):
        pass

    def process(self):
        data_path = self.data_path
        phase = self.phase
        data_list= []

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'B': 5, 'Si': 6, 'Ru': 7, 'As': 8, 'Pt': 9, 'Fe': 10, 'Se': 11,
                 'S': 12, 'Cl': 13, 'I': 14, 'Br': 15, 'P': 16}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        bind_df = pd.read_csv(data_path / f"{phase}.csv")


        for _, i in bind_df.iterrows():
            smiles = i["SMILES"]
            # print(smiles)
            try:
                rdkit_mol = AllChem.MolFromSmiles(smiles)
                if rdkit_mol != None:  # ignore invalid mol objects
                    data = mol_to_graph_data_obj_simple(rdkit_mol)

                    mol = Chem.MolFromSmiles(smiles)
                    N = mol.GetNumAtoms()
                    type_idx = []
                    atomic_number = []
                    aromatic = []
                    sp = []
                    sp2 = []
                    sp3 = []
                    num_hs = []
                    for atom in mol.GetAtoms():
                        type_idx.append(types[atom.GetSymbol()])
                        atomic_number.append(atom.GetAtomicNum())
                        aromatic.append(1 if atom.GetIsAromatic() else 0)
                        hybridization = atom.GetHybridization()
                        sp.append(1 if hybridization == HybridizationType.SP else 0)
                        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

                    z = torch.tensor(atomic_number, dtype=torch.long)


                    row, col, edge_type = [], [], []
                    for bond in mol.GetBonds():
                        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                        row += [start, end]
                        col += [end, start]
                        edge_type += 2 * [bonds[bond.GetBondType()]]

                    edge_index = torch.tensor([row, col], dtype=torch.long)

                    edge_type = torch.tensor(edge_type, dtype=torch.long)

                    edge_attr = F.one_hot(edge_type,
                                          num_classes=len(bonds)).to(torch.float)

                    perm = (edge_index[0] * N + edge_index[1]).argsort()
                    edge_index = edge_index[:, perm]
                    edge_type = edge_type[perm]
                    edge_attr = edge_attr[perm]

                    row, col = edge_index
                    hs = (z == 1).to(torch.float)
                    num_hs = scatter(hs[row], col, dim_size=N).tolist()

                    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
                    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                                      dtype=torch.float).t().contiguous()
                    x = torch.cat([x1.to(torch.float), x2], dim=-1)

                    e_attr = [bonds_dec[int(ii)] for ii in edge_attr.argmax(dim=1)]
                    n_attr = {kk: qm9_id2ele[int(ii)] for kk, ii in enumerate(x[:, :5].argmax(dim=1))}

                    nxG = to_network(edge_index, e_attr, n_attr)
                    known = set([])
                    mp_edges = set([])
                    for edges in nxG.edges:
                        e1, e2 = tuple(edges)

                        e1_nei = nx.neighbors(nxG, e1)
                        e2_nei = nx.neighbors(nxG, e2)
                        for n1 in e1_nei:
                            fmp = (n1, e1, e2)
                            if (fmp not in known) and (fmp[::-1] not in known):
                                known.add(fmp)
                                known.add(fmp[::-1])

                                mp = [nxG.nodes[fmp[0]]['ele'], nxG[fmp[0]][fmp[1]]['attr'],
                                      nxG.nodes[fmp[1]]['ele'], \
                                      nxG[fmp[1]][fmp[2]]['attr'], nxG.nodes[fmp[2]]['ele']]
                                mp = ''.join(mp)
                                if mp in MP_corpus:
                                    mp_edges.add((e1, e2, MP2id[mp], 3))
                        for n2 in e2_nei:
                            fmp = (e1, e2, n2)
                            if (fmp not in known) and (fmp[::-1] not in known):
                                known.add(fmp)
                                known.add(fmp[::-1])
                                mp = [nxG.nodes[fmp[0]]['ele'], nxG[fmp[0]][fmp[1]]['attr'],
                                      nxG.nodes[fmp[1]]['ele'], \
                                      nxG[fmp[1]][fmp[2]]['attr'], nxG.nodes[fmp[2]]['ele']]
                                mp = ''.join(mp)
                                if mp in MP_corpus:
                                    mp_edges.add((e1, e2, MP2id[mp], 3))
                        for n1, n2 in zip(e1_nei, e2_nei):
                            fmp = (n1, e1, e2, n2)
                            if (fmp not in known) and (fmp[::-1] not in known):
                                known.add(fmp)
                                known.add(fmp[::-1])
                                mp = [nxG.nodes[fmp[0]]['ele'], nxG[fmp[0]][fmp[1]]['attr'],
                                      nxG.nodes[fmp[1]]['ele'], \
                                      nxG[fmp[1]][fmp[2]]['attr'], nxG.nodes[fmp[2]]['ele'],
                                      nxG[fmp[2]][fmp[3]]['attr'], \
                                      nxG.nodes[fmp[3]]['ele']]
                                mp = ''.join(mp)
                                if mp in MP_corpus:
                                    mp_edges.add((n1, n2, MP2id[mp], 4))

                    if len(mp_edges) > 0:
                        mp_edges = [list(i) for i in mp_edges]
                        mp_add = torch.tensor(mp_edges)

                        edg_add = mp_add[:, :2].T
                        edge_index = torch.cat([edge_index, edg_add], dim=1)
                        klp = mp_add[:, 2] + edge_attr.size(1)

                        edge_attr = edge_attr.argmax(dim=1)
                        edge_attr = torch.cat([edge_attr, klp], dim=0)

                    else:
                        edge_attr = edge_attr.argmax(dim=1)

                    data.meth_index = torch.LongTensor(edge_index)
                    data.meth_attr = torch.LongTensor(edge_attr)
                    data.meth_x = x

                    seq = i["Target_Sequence"]
                    target = seq_cat(seq)
                    labels = i["Label"]
                    smiles = label_smiles(smiles, 150)
                    # target = seq_cat(protein[key])
                    data.target = torch.LongTensor([target])

                    data.smiles = torch.Tensor([smiles])

                    data.y = torch.Tensor(([labels]))

                    data_list.append(data)
            except:
                continue

        self.mydata = data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

