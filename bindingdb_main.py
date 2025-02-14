import argparse
import torch
from tqdm import tqdm
from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from BindDb_dataset import MoleculeDataset
import time
from datetime import datetime
from pathlib import Path
import h5py
from torch.cuda.amp import autocast, GradScaler
from math import exp
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch_geometric.data import InMemoryDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models import motif_affinity, test


def main():
    # Training settings
    SHOW_PROCESS_BAR = True
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')

    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='sider',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='../saved_model/pretrained.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    data_path = '../Binddb_data/'
    root = '../datalatest/'

    batch_size = 32
    n_epoch = 20
    interrupt = None
    save_best_epoch = 13  # when `save_best_epoch` is reached and the loss starts to decrease, save best model parameters

    seed = np.random.randint(33927, 33928)  ##random
    path = Path(f'runs/BindingDB_1000_150{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(path)
    f_param = open(path / 'parameters.txt', 'w')

    print(f'device={device}')
    print(f'seed={seed}')
    print(f'write to {path}')
    f_param.write(f'device={device}\n'
                  f'seed={seed}\n'
                  f'write to {path}\n')

    # set up model
    motif_embeding = GNN_graphpred(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                                   graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == "":
        motif_embeding.from_pretrained(args.input_model_file)
    model = motif_affinity(args.num_layer, args.emb_dim, motif_embeding, JK=args.JK, drop_ratio=args.dropout_ratio)

    train_data = MoleculeDataset(root, data_path, phase='train')
    validation_data = MoleculeDataset(root, data_path, phase='valid')
    test_data = MoleculeDataset(root, data_path, phase='test')
    train_loader = DataLoader(train_data.mydata, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_data.mydata, batch_size=batch_size, pin_memory=True, shuffle=True,
                                   drop_last=True)
    test_loader = DataLoader(test_data.mydata, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=n_epoch,
                                              steps_per_epoch=len(train_loader))
    loss_function = nn.CrossEntropyLoss()

    scaler = GradScaler()

    start = datetime.now()
    print('start at ', start)
    model.to(device)

    best_epoch = -1
    best_val_loss = 0
    losses = []
    for epoch in range(1, n_epoch + 1):
        tbar = tqdm(enumerate(train_loader), disable=not SHOW_PROCESS_BAR, total=len(train_loader))
        for batch_idx, data in tbar:
            model.train()
            data = data.to(device)

            optimizer.zero_grad()
            with autocast():
                # constr_loss, output = model(data)
                loss = model(data)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() / len(data.y):.3f}')

        performance_train = test(model, train_loader, loss_function, device, False)

        for i in performance_train:
            writer.add_scalar(f'training {i}', performance_train[i], global_step=epoch)

        performance_validation = test(model, validation_loader, loss_function, device, False)
        for i in performance_validation:
            writer.add_scalar(f'validation {i}', performance_validation[i], global_step=epoch)

        losses.append(loss.item() / len(data.y))

        if epoch >= save_best_epoch and performance_validation['AUC'] > best_val_loss:
            best_val_loss = performance_validation['AUC']
            best_epoch = epoch
            torch.save(model.state_dict(), path / 'best_model.pt')

    model.load_state_dict(torch.load(path / 'best_model.pt'))
    with open(path / 'result.txt', 'w') as f:
        f.write(f'best model found at epoch NO.{best_epoch}\n')
        performance_train = test(model, train_loader, loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'training:\n')
        print(f'training:')
        for k, v in performance_train.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')

        performance_validation = test(model, validation_loader, loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'validation:\n')
        print(f'validation:')
        for k, v in performance_validation.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')

        performance_test = test(model, test_loader, loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'test:\n')
        print(f'test:')
        for k, v in performance_test.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')

    print('training finished')

    end = datetime.now()
    print('end at:', end)
    print('time used:', str(end - start))


if __name__ == "__main__":
    main()


