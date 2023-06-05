import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_cluster import knn_graph

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = torch.matmul(edge_index, x)
        x = self.lin2(x)
        return x

def train(data_loader):
    model.train()
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def test(data_loader):
    model.eval()
    correct = 0
    for data in data_loader:
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    return correct / len(data.test_mask)

# 模拟数据
num_nodes = 50000
num_features = 10
x = torch.randn((num_nodes, num_features))
y = torch.randint(0, 2, (num_nodes,))
y[y==0] = -1 # 将标签为0的客户设置为-1

# 构建图
edge_index = knn_graph(x, k=6) # 前6个相似度最高

# 初始化模型和优化器
model = GCN(num_features=num_features, hidden_channels=16, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 训练和测试模型
for epoch in range(1, 201):
    train(data_loader)
    test_acc = test(data_loader)
    print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')