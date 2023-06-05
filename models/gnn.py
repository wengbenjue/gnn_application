import torch
from models.gat import GAT
from models.graphsage import GraphSage
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
# PRW：https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
# from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)  # GraphSage / GAT
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'

        for l in range(args.num_layers - 1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

            # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            '''
            当使用num heads>1应用GAT时，需要修改conv层的输入和输出维度（self.convs），确保下一层的输入dim为num_heads乘以上一层的输出尺寸。
提示：如果想实现多头（multi-heads），需要在建立self.convs时改变self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim))和post-message-passing中第一个nn.Linear(hidden_dim * num_heads, hidden_dim) 
         '''
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        # Applies a softmax followed by a logarithm.
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.log_softmax
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        # The negative log likelihood loss.
        # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#nll_loss
        return F.nll_loss(pred, label)