# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification


# 定义GAT图卷积层
class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.0):
        super(GATConv, self).__init__()
        self.in_channels = in_channels  # 输入特征维度
        self.out_channels = out_channels  # 输出特征维度
        self.heads = heads  # 注意力头数
        self.concat = concat  # 是否拼接多头输出
        self.negative_slope = negative_slope  # leakyrelu的负斜率
        self.dropout = dropout  # dropout概率

        # 定义可训练的参数，线性变换矩阵和注意力向量
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        # 初始化参数
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        # x: 节点特征矩阵，大小为 [num_nodes, in_channels]
        # edge_index: 边索引矩阵，大小为 [2, num_edges]

        num_nodes = x.size(0)  # 节点数

        # 线性变换节点特征
        x = torch.mm(x, self.weight)  # 大小为 [num_nodes, heads * out_channels]
        x = x.view(-1, self.heads, self.out_channels)  # 大小为 [num_nodes, heads, out_channels]

        # 计算注意力系数
        x_i = x[edge_index[0]]  # 大小为 [num_edges, heads, out_channels]
        x_j = x[edge_index[1]]  # 大小为 [num_edges, heads, out_channels]
        alpha = torch.cat([x_i, x_j], dim=-1)  # 大小为 [num_edges, heads, 2 * out_channels]
        alpha = (alpha * self.att).sum(dim=-1)  # 大小为 [num_edges, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)  # 大小为 [num_edges, heads]

        # 归一化注意力系数
        alpha = alpha.view(-1, self.heads)  # 大小为 [num_edges * heads]
        alpha = F.softmax(alpha, dim=0)  # 大小为 [num_edges * heads]

        # dropout操作
        alpha = F.dropout(alpha, p=self.dropout)  # 大小为 [num_edges * heads]

        # 加权聚合邻居特征
        alpha = alpha.view(-1)[None, :]  # 大小为 [1,num_edges * heads]
        x_j = x_j.view(-1, self.heads * self.out_channels)  # 大小为 [num_edges ,heads*out_channels]

        out = torch.sparse.mm(edge_index.t(), alpha.t() * x_j)  # 大小为 [num_nodes ,heads*out_channels]

        if self.concat:
            out = out.view(num_nodes, -1)  # 大小为 [num_nodes ,heads*out_channels]
            return out

        else:
            out_mean = out.view(num_nodes, self.heads, self.out_channels).mean(dim=1)  # 大小为 [num_nodes ,out_channels]
            return out_mean


# 定义GAT模型
class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

        self.gatconv1 = GATConv(in_channels=10, out_channels=8)

        self.gatconv2 = GATConv(in_channels=8, out_channels=2)

    def forward(self, x):
        num_nodes = x.size(0)  # 节点数
        edge_index = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=torch.long).repeat(1,
                                                                                                               num_nodes)
        edge_index = torch.cat([edge_index, (edge_index + 10) % 10], dim=0)

        x = self.gatconv1(x=x, edge_index=edge_index)
        x = F.relu(x)

        x = self.gatconv2(x=x, edge_index=edge_index)

        return F.log_softmax(x, dim=-1)


# 模拟数据集，每个客户有10维特征，标签为0或1，其中标签为0的客户比较少
X, y = make_classification(n_samples=50000, n_features=10, n_classes=2, n_informative=8, n_redundant=0, n_repeated=0,
                           class_sep=3.0, n_clusters_per_class=2,
                           weights=[0.05], flip_y=0.01)

# 将数据转换成Pytorch张量，并划分训练集和测试集
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

train_X = X[:40000]
train_y = y[:40000]

test_X = X[40000:]
test_y = y[40000:]

# 定义超参数和优化器
batch_size = 128
epochs = 10

model = GAT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(epochs):
    model.train()
    loss_all = 0

    for i in range(0, len(train_X), batch_size):
        batch_X = train_X[i:i + batch_size]
        batch_y = train_y[i:i + batch_size]

        optimizer.zero_grad()

        output = model(batch_X)

        loss = F.nll_loss(output, batch_y)

        loss.backward()

        loss_all += loss.item() * batch_size

        optimizer.step()

    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch + 1, (loss_all / len(train_X))))

# 测试模型
model.eval()
correct = 0

for i in range(0, len(test_X), batch_size):
    batch_X = test_X[i:i + batch_size]
    batch_y = test_y[i:i + batch_size]

    output = model(batch_X)

    pred = output.max(dim=-1)[1]

    correct += pred.eq(batch_y).sum().item()

print('Accuracy: {:.4f}'.format(correct / len(test_X)))