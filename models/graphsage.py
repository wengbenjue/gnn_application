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
class GraphSage(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=True,
                 bias=False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = None
        self.lin_r = None

        ############################################################################
        # 定义下面的message和update 函数所需的层。
        # self.lin_l 是应用于中心节点嵌入的线性变换。
        # self.lin_r 是应用于来自邻居的聚合message的线性变换。

        self.lin_l = Linear(in_channels, out_channels)  # Wl
        self.lin_r = Linear(in_channels, out_channels)  # Wr

        ############################################################################

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        """"""

        out = None

        ############################################################################
        # 实现消息传递以及任何后处理（更新规则）
        # 1. 首先调用propagate函数来进行消息传递。
        #    1.1 有关更多信息，请参见此处:
        #        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        #    1.2 对中心（x_central）和邻居（x_neighbor）节点使用相同的表示，即x=（x，x）进行传播。
        # 2. 使用skip connection更新节点嵌入。
        # 3. 如果需要归一化, 使用L-2 normalization (定义在torch.nn.functional)

        out = self.propagate(edge_index, x=(x, x), size=size)
        x = self.lin_l(x)
        out = self.lin_r(out)
        out = out + x
        if self.normalize:
            out = F.normalize(out)

        ############################################################################

        return out

    def message(self, x_j):
        out = None

        # propagte传入的中心节点和邻居节点的表示一样。
        out = x_j

        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = None

        # 沿其索引节点数的维度.
        node_dim = self.node_dim

        ############################################################################
        # 实现平均聚合.
        # 请参见此处，了解如何使用torch_scatter.scatter:
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter

        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='mean')

        ############################################################################

        return out