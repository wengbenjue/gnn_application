import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


#训练、测试
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

# from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt
from models.gnn import GNNStack

def train(dataset,train_loader, test_loader, args):
    # print("节点分类任务，数据集大小:", np.sum(dataset[0]['train_mask'].numpy()))
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # loader = train_loader

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
                     args)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_accs = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accs.append(test_acc)
            # print(test_acc)
            print("loss: {0},Val Acc: {1}".format(total_loss,test_acc))
        else:
            test_accs.append(test_accs[-1])
    return test_accs, losses


def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]

        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
    return correct / total


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def main():
    for args in [
        {'model_type': 'GAT',
         'dataset': 'cora',
         # 'dataset': 'custom',
         'num_layers': 2,
         'heads': 1,
         'batch_size': 32,
         'hidden_dim': 32,
         'dropout': 0.5,
         'epochs': 700,
         'opt': 'adam',
         'opt_scheduler': 'none',
         'opt_restart': 0,
         'weight_decay': 5e-3,
         'lr': 0.01
         },
    ]:
        args = objectview(args)
        for model in ['GraphSage', 'GAT']:
            args.model_type = model

            # Match the dimension.
            if model == 'GAT':
              args.heads = 2
            else:
              args.heads = 1

            if args.dataset == 'cora':
                dataset = Planetoid(root='./cora', name='Cora')
            elif args.dataset == 'custom':
                from torch_geometric.data import Data

                # data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
                # 也可以以该矩阵的转置形式定义edge_index
                edge_index = torch.tensor([[0, 1, 1, 2],
                                           [1, 0, 2, 1]], dtype=torch.long)

                # data.x: Node feature matrix with shape [num_nodes, num_node_features]
                x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

                dataset = Data(x=x, edge_index=edge_index)
                print(dataset)
            else:
                print()
                # raise NotImplementedError("Unknown dataset")

        test_accs, losses = train(dataset, train_loader=None, test_loader=None, args=args)
        print("Maximum accuracy: {0}".format(max(test_accs)))
        print("Minimum loss: {0}".format(min(losses)))

        plt.title(dataset.name)
        plt.plot(losses, label="training loss" + " - " + args.model_type)
        plt.plot(test_accs, label="test accuracy" + " - " + args.model_type)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
