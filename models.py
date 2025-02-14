import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn.conv import GATConv,GCNConv
from torch.nn import Sequential, Linear, ReLU, GRU,Dropout
from torch_geometric.nn import NNConv, Set2Set
from tqdm.auto import tqdm
import numpy as np
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from sklearn.metrics import roc_auc_score, precision_score, recall_score,f1_score


device = torch.device("cuda:0")
EPS = 1e-10

def T2LKL(p, q):
    kl_div = F.kl_div(q.log() + EPS, p, reduction='batchmean')
    return kl_div

def Lalign(X_q, h_readout, z_readout, T=4):
    loss = X_q * sum(T2LKL(F.softmax(h, dim=0), F.softmax(z, dim=0)) for h, z in zip(h_readout, z_readout))
    return loss

class MutualAttentation(nn.Module):
    def __init__(self, in_channels=128, att_size=128, c_m=128, c_n=8):
        super().__init__()
        self.bipool = BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linearS = nn.Linear(in_channels, att_size)
        self.linearT = nn.Linear(in_channels, att_size)

    def forward(self, source, target):
        '''
        source: (batch, channels, seq_len)
        target: (batch, channels, seq_len)
        global_descriptor: (batch, 1, channels)
        '''
        global_descriptor = self.bipool(source)
        target_org = target
        target = self.linearT(target.permute(0, 2, 1)).permute(0, 2, 1)
        global_descriptor = self.linearS(global_descriptor)
        att_maps = torch.bmm(global_descriptor, target)
        att_maps = F.sigmoid(att_maps)
        out_target = torch.add(target_org, torch.mul(target_org, att_maps))
        out_target = F.relu(out_target)

        return out_target


def getMaskedEdges(edge_index, edge_attr, eType):
    return edge_index


class BilinearPooling(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, c_m=128, c_n=8):
        super().__init__()

        self.convA = torch.nn.Conv1d(in_channels, c_m, kernel_size=1, stride=1, padding=0)
        self.convB = torch.nn.Conv1d(in_channels, c_n, kernel_size=1, stride=1, padding=0)
        self.linear = torch.nn.Linear(c_m, out_channels, bias=True)

    def forward(self, x):
        A = self.convA(x)
        B = self.convB(x)
        att_maps = F.softmax(B, dim=-1)
        global_descriptors = torch.bmm(A, att_maps.permute(0, 2, 1))
        global_descriptor = torch.mean(global_descriptors, dim=-1)
        out = self.linear(global_descriptor).unsqueeze(1)

        return out


class MutualAttentation(nn.Module):
    def __init__(self, in_channels=128, att_size=128, c_m=128, c_n=8):
        super().__init__()
        self.bipool = BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linearS = torch.nn.Linear(in_channels, att_size)
        self.linearT = torch.nn.Linear(in_channels, att_size)

    def forward(self, source, target):
        global_descriptor = self.bipool(source)
        target_org = target
        target = self.linearT(target.permute(0, 2, 1)).permute(0, 2, 1)
        global_descriptor = self.linearS(global_descriptor)
        att_maps = torch.bmm(global_descriptor, target)
        att_maps = F.sigmoid(att_maps)
        out_target = torch.add(target_org, torch.mul(target_org, att_maps))
        out_target = F.relu(out_target)

        return out_target


class RelationalAttention(nn.Module):
    def __init__(self, in_feat, edges_num, activation_fc):
        super(RelationalAttention, self).__init__()

        self.edges_num = edges_num
        self.importance_fn = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=2 * in_feat, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2 * in_feat, out_features=1, bias=False)
        )
        self.activatin_fc = activation_fc

    def forward(self, emb):
        rela_importance = self.importance_fn(emb)
        rela_importance = torch.mean(rela_importance, dim=0)
        rela_weights = torch.softmax(rela_importance, dim=0)
        nodes_embs = emb * rela_weights
        nodes_embs_out = nodes_embs.sum(1)
        return nodes_embs_out


class MolAtt(nn.Module):
    def __init__(self, in_feat, out_feat, etypes, n_heads, dropout=0.2, activation_fc=None, residual=True):
        super(MolAtt, self).__init__()

        self.etypes = etypes
        self.residual = residual

        self.multi_gat_layers = nn.ModuleList()
        for etype in range(self.etypes):
            self.multi_gat_layers.append(GATConv(in_feat, out_feat, heads=n_heads, concat=True))
        self.relaAttnlayer = RelationalAttention(in_feat=out_feat * n_heads, edges_num=self.etypes,
                                                 activation_fc=activation_fc)
        self.dropout = nn.Dropout(dropout)
        self.activation_fc = activation_fc
        self.batchnorm_h = nn.BatchNorm1d(out_feat * n_heads)

    def forward(self, h, batch):
        emb = h
        output = []
        for etype in range(self.etypes):
            masked_edgeIndex = getMaskedEdges(batch.edge_index, batch.edge_attr[:self.etypes], eType=etype)
            etype_h = self.multi_gat_layers[etype](x=emb, edge_index=masked_edgeIndex.to(torch.long))
            output.append(etype_h)
        output = torch.stack(output, 1)
        output = self.relaAttnlayer(output)
        output = F.elu(output)
        return output

pretrained_embeddings = np.load('1mer.npy')

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class motif_affinity(torch.nn.Module):
    def __init__(self, num_layer, emb_dim,  smimodel, JK="last",drop_ratio=0 ):
        super(motif_affinity, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.motif_emd = smimodel

        self.embedding_weights = torch.tensor(pretrained_embeddings, dtype=torch.float32)
        self.embedding_xt = torch.nn.Embedding.from_pretrained(self.embedding_weights)
        self.embedding_xt.weight.requires_grad = True

        # self.embedding_xt = torch.nn.Linear(21, 128)
        self.smi_emd = torch.nn.Linear(300,128)

        self.GRU_xt_1 = torch.nn.GRU(128, 64, 2, batch_first=True, bidirectional=True)
        self.fc_xt = torch.nn.Linear(1000 * 128, 128)

        self.fc1 = torch.nn.Linear(256, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.out = torch.nn.Linear(512, 1)
        self.out = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

        self.liner1 = torch.nn.Linear(128, 64)
        self.liner2 = torch.nn.Linear(128, 64)
        self.liner3 = torch.nn.Linear(64, 128, bias=False)
        self.liner4 = torch.nn.Linear(64, 128, bias=False)
        self.mut_att = MutualAttentation()


        #MpMOl
        infeat=23
        dim=64
        edge_dim=1
        nheads=8
        dropout=0.2
        recurs=3

        self.lin0 = torch.nn.Linear(infeat, dim)
        self.layers = torch.nn.ModuleList()
        self.edge_dim = edge_dim  # 1
        self.recurs = recurs
        nn = Sequential(
            Linear(self.edge_dim, 128),
            ReLU(),
            Linear(128, dim * dim)
        )
        self.dropout = Dropout(dropout)
        self.layers.append(MolAtt(in_feat=dim, out_feat=int(dim / nheads), etypes=self.edge_dim,
                                  n_heads=nheads, dropout=dropout, activation_fc=F.relu))
        self.layers.append(NNConv(dim, dim, nn, aggr='mean'))
        self.gru = GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=5)
        # self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin1 = torch.nn.Linear(2 * dim, 128)
        self.lin2 = torch.nn.Linear(dim, 1)
        self.relu = torch.nn.ReLU()

        self.bcn = weight_norm(
            BANLayer(v_dim=128, q_dim=128, h_dim=256, h_out=2),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(512, 512, 128, binary=2)
        self.smi_embed = torch.nn.Embedding(53, 128)
        self.bilstm = torch.nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        target = data.target

        smiles = data.smiles
        smiles = smiles.long()
        smi_embed = self.smi_embed(smiles)
        xd_lstm, _ = self.bilstm(smi_embed)  # (N,L,128)

        embedded_xt = self.embedding_xt(target)
        gru_xt, h0 = self.GRU_xt_1(embedded_xt)

        xt = gru_xt.contiguous().view(-1, 1000 * 128)
        xt = self.fc_xt(xt)

        constr_loss,node_representation = self.motif_emd(x, edge_index, edge_attr)

        f, att = self.bcn(xd_lstm, gru_xt)

        x = gap(node_representation, batch)
        x = self.smi_emd(x)

        # meth

        data.meth_x = data.meth_x.to(torch.float32)
        out = F.relu(self.lin0(data.meth_x))
        h = out.unsqueeze(0)
        # data.edge_index = data.edge_index.to(torch.float32)
        data.meth_attr = data.meth_attr.to(torch.float32)
        for recur in range(self.recurs):
            out = F.relu(self.layers[0](out, data))
            out = self.dropout(out)
            data.meth_attr = data.meth_attr.unsqueeze(1)
            m = F.relu(self.layers[1](out, data.meth_index, data.meth_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)
        xd_gra = self.relu(out)  # (batchsize,128)

        xd_motif = self.liner1(x)
        xd_meth = self.liner2(xd_gra)
        xd_motif = self.relu(xd_motif)
        xd_meth = self.relu(xd_meth)
        xd_motif = self.liner3(xd_motif)
        xd_meth = self.liner4(xd_meth)
        score = torch.sigmoid(xd_motif + xd_meth)
        xd = score * xd_motif + (1 - score) * xd_meth

        # # concat
        xc = torch.cat((xd,xt), 1)

        out = torch.cat((xc,f),1)
        score = self.mlp_classifier(out)
        return score



    def __call__(self, data, train=True):
        correct_interaction = data.y
        correct_interaction = correct_interaction.long()
        Loss = nn.CrossEntropyLoss()
        if train:
            predicted_interaction = self.forward(data)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            correct_interaction = data.y

            predicted_interaction = self.forward(data)
            # correct_labels = correct_interaction.to('cpu').data.numpy().item()
            correct_labels = correct_interaction.to('cpu').numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[0, 1]

            return correct_labels, predicted_labels, predicted_scores


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)
        self.bn3 = torch.nn.BatchNorm1d(out_dim)
        self.fc4 = torch.nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


def test(model, test_loader, loss_function, device, show=True):
    model.eval()
    T, Y, S = [], [], []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            data = data.to(device)
            correct_labels, predicted_labels, predicted_scores = model(data, train=False)
            T.extend(correct_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)
    T = np.array(T)
    Y = np.array(Y)
    S = np.array(S)

    print(f'T shape: {T.shape}')
    print(f'Y shape: {Y.shape}')

    AUC = roc_auc_score(T, S)
    precision = precision_score(T, Y)
    recall = recall_score(T, Y)
    f1 = f1_score(T, Y)
    evaluation = {
        'AUC': AUC,
        'precision': precision,
        'recall': recall,
        'F1 ':f1}

    return evaluation


if __name__ == "__main__":
    pass

