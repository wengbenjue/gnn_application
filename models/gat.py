import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter, Linear
import torch_scatter
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
class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=2,
                 negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        ############################################################################
        # self.lin_l是在消息传递之前应用于嵌入的线性变换。
        # 注意线性层的尺寸，因为我们使用的是多头注意力。

        self.lin_l = Linear(in_channels, heads * out_channels)

        ############################################################################

        self.lin_r = self.lin_l  # W_r

        ############################################################################
        # 定义注意力参数，需要考虑多头的情况
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        ############################################################################

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_

    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels

        ############################################################################
        # 主要逻辑实现函数，实现消息传递、预处理、后处理
        # 1. 首先对节点嵌入应用线性变换，并将其拆分为多头。对源节点和目标节点使用相同的表示，但应用不同的线性权重（W_l和W_r）
        # 2. 计算中心节点（alpha_l）和相邻节点（alpha_r）的alpha向量
        # 3. 调用propagate函数进行消息传递。使得alpha = (alpha_l, alpha_r)传递参数。
        # 4. 将输出转换回N*d的形状。

        x_l = self.lin_l(x)
        x_r = self.lin_r(x)
        x_l = x_l.view(-1, H, C)
        x_r = x_r.view(-1, H, C)
        alpha_l = (x_l * self.att_l).sum(axis=1)  # *是逐元素相乘
        alpha_r = (x_r * self.att_r).sum(axis=1)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        out = out.view(-1, H * C)
        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        ############################################################################
        # 实现message功能。将注意力放在message中
        # 1. 使用alpha_i和alpha_j计算最终注意力权重，并应用leaky Relu。
        # 2. 为所有节点计算邻居节点上的softmax。使用torch_geometric.utils.softmax而不是pytorch中的softmax。
        # 3. 对注意权重（alpha）应用dropout。
        # 4. 增加嵌入和注意力权重，输出应为形状 E * H * d。
        # 5. ptr (LongTensor, 可选):如果给定，则根据CSR表示中的排序输入计算softmax。

        # alpha：[E, C]
        alpha = alpha_i + alpha_j  # leaky_relu的对象
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch-geometric-utils
        # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/utils/softmax.py

        alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(1)  # [E,1,C]
        out = x_j * alpha  # [E,H,C]

        ############################################################################

        return out

    def aggregate(self, inputs, index, dim_size=None):
        ############################################################################
        # 实现聚合函数
        # 请参见此处，了解如何使用 torch_scatter.scatter: https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/scatter.html
        # 请注意“reduce”参数与GraphSAGE中的参数不同

        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')

        ############################################################################

        return out