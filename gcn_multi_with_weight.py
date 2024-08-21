import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import add_self_loops, degree, subgraph
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt


def generate_and_save_data(num_nodes=100, num_edges=500, node_feature_path="node_features.npy",
                           edge_data_path="edge_data.npy"):
    """
    生成模拟数据，包括节点特征和边的权重，并保存到文件。

    参数:
    - num_nodes: 节点数量
    - num_edges: 边的数量
    - node_feature_path: 节点特征的保存路径
    - edge_data_path: 边数据（索引和权重）的保存路径
    """
    # 生成随机节点特征，假设每个节点有16维特征
    node_features = np.random.randn(num_nodes, 16)
    np.save(node_feature_path, node_features)

    # 生成随机边，边的索引和权重
    edge_index = np.random.randint(0, num_nodes, (2, num_edges))
    edge_weight = np.random.rand(num_edges) * 100
    edge_data = np.vstack([edge_index, edge_weight])
    np.save(edge_data_path, edge_data)

    print(f"Node features saved to {node_feature_path}")
    print(f"Edge data (index and weight) saved to {edge_data_path}")


# 从文件加载节点特征和边的权重，并构建edge_index
def load_data(node_feature_path="node_features.npy", edge_data_path="edge_data.npy"):
    """
    从文件加载节点特征和边数据，并构建 PyG 数据对象。

    参数:
    - node_feature_path: 节点特征的文件路径
    - edge_data_path: 边数据的文件路径

    返回:
    - PyG 数据对象
    """
    if os.path.exists(node_feature_path) and os.path.exists(edge_data_path):
        # 加载节点特征和边数据
        node_features = np.load(node_feature_path)
        edge_data = np.load(edge_data_path)

        # 转换为 PyTorch 张量
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_data[:2, :], dtype=torch.long)
        edge_weight = torch.tensor(edge_data[2, :], dtype=torch.float)

        # 随机生成节点标签（3个类别）
        y = torch.randint(0, 3, (x.size(0),))

        # 构建 PyG 数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        print(f"Data loaded from {node_feature_path} and {edge_data_path}")
    else:
        print(f"File(s) not found. Generating new data.")
        generate_and_save_data(node_feature_path=node_feature_path, edge_data_path=edge_data_path)
        data = load_data(node_feature_path, edge_data_path)

    return data


# 定义 GCN 模型
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化 GCN 模型。

        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层特征维度
        - output_dim: 输出类别数
        """
        super(GCNModel, self).__init__()
        # 定义两个 GCN 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        """
        前向传播函数。

        参数:
        - data: PyG 数据对象

        返回:
        - 模型的输出
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # 添加自环（以防止节点没有邻居）
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=data.num_nodes)
        row, col = edge_index
        # 计算节点的度数
        deg = degree(row, data.num_nodes, dtype=x.dtype)
        # 计算度数的逆平方根
        deg_inv_sqrt = deg.pow(-0.5)
        # 计算归一化系数
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # 第一层卷积
        x = self.conv1(x, edge_index, norm)
        x = F.relu(x)
        # 第二层卷积
        x = self.conv2(x, edge_index, norm)
        return F.log_softmax(x, dim=1)


# 定义 GraphSAGE 模型
class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化 GraphSAGE 模型。

        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层特征维度
        - output_dim: 输出类别数
        """
        super(GraphSAGEModel, self).__init__()
        # 定义两个 SAGE 层
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        """
        前向传播函数。

        参数:
        - data: PyG 数据对象

        返回:
        - 模型的输出
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # 添加自环（以防止节点没有邻居）
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=data.num_nodes)
        # 第一层卷积
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        # 第二层卷积
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


# 定义 GAT 模型
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        """
        初始化 GAT 模型。

        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层特征维度
        - output_dim: 输出类别数
        - num_heads: 注意力头的数量
        """
        super(GATModel, self).__init__()
        # 定义 GAT 层
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)

    def forward(self, data):
        """
        前向传播函数。

        参数:
        - data: PyG 数据对象

        返回:
        - 模型的输出
        """
        x, edge_index = data.x, data.edge_index
        # 第一层 GAT 卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层 GAT 卷积
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 训练函数
def train(model, data, optimizer, criterion, device, epochs=200):
    """
    训练模型。

    参数:
    - model: 模型对象
    - data: PyG 数据对象
    - optimizer: 优化器
    - criterion: 损失函数
    - device: 设备（CPU 或 GPU）
    - epochs: 训练轮数

    返回:
    - 训练过程中的损失值列表
    """
    model.train()
    train_losses = []
    data = data.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return train_losses


# 测试函数
def test(model, data, device):
    """
    测试模型。

    参数:
    - model: 模型对象
    - data: PyG 数据对象
    - device: 设备（CPU 或 GPU）

    返回:
    - 准确率、精确率、召回率、F1分数
    """
    model.eval()
    data = data.to(device)
    out = model(data)
    _, pred = out.max(dim=1)
    acc = accuracy_score(data.y.cpu(), pred.cpu())
    precision, recall, f1, _ = precision_recall_fscore_support(data.y.cpu(), pred.cpu(), average='weighted')

    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return acc, precision, recall, f1


# 可视化训练损失
def plot_losses(train_losses):
    """
    绘制训练损失曲线。

    参数:
    - train_losses: 训练过程中的损失值列表
    """
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# 保存模型
def save_model(model, path='gcn_model.pth'):
    """
    保存模型参数到文件。

    参数:
    - model: 模型对象
    - path: 保存路径
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


# 加载模型
def load_model(model, path='gcn_model.pth', device='cpu'):
    """
    从文件加载模型参数。

    参数:
    - model: 模型对象
    - path: 模型文件路径
    - device: 设备（CPU 或 GPU）
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f'Model loaded from {path}')


# 预测函数
def predict(model, data, device):
    """
    使用模型进行预测。

    参数:
    - model: 模型对象
    - data: PyG 数据对象
    - device: 设备（CPU 或 GPU）

    返回:
    - 预测结果
    """
    model.eval()
    data = data.to(device)
    out = model(data)
    _, pred = out.max(dim=1)
    return pred.cpu().numpy()


# 筛选子图函数
def filter_data_by_mask(data, mask):
    """
    根据掩码筛选数据子集。

    参数:
    - data: PyG 数据对象
    - mask: 节点掩码

    返回:
    - 筛选后的 PyG 数据对象
    """
    sub_edge_index, sub_edge_attr = subgraph(mask, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True)
    return Data(x=data.x[mask], edge_index=sub_edge_index, edge_attr=sub_edge_attr, y=data.y[mask])


# 主函数
def main(model_type='GCN'):
    """
    主函数，用于选择模型类型、训练、测试和评估模型。

    参数:
    - model_type: 选择的模型类型（'GCN'、'GraphSAGE'、'GAT'、'HAT'）
    """
    # 选择设备（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载（如果不存在则生成并保存）
    data = load_data()

    # 数据集划分（训练集和测试集）
    train_mask, test_mask = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)
    train_data = filter_data_by_mask(data, train_mask)
    test_data = filter_data_by_mask(data, test_mask)

    input_dim = data.num_node_features
    hidden_dim = 32
    output_dim = 3

    # 根据指定的模型类型选择模型
    if model_type == 'GCN':
        model = GCNModel(input_dim, hidden_dim, output_dim).to(device)
    elif model_type == 'GraphSAGE':
        model = GraphSAGEModel(input_dim, hidden_dim, output_dim).to(device)
    elif model_type == 'GAT':
        model = GATModel(input_dim, hidden_dim, output_dim).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(model)

    # 选择优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # 训练模型
    train_losses = train(model, train_data, optimizer, criterion, device)

    # 测试模型
    print("Training Set Evaluation:")
    train_acc, train_precision, train_recall, train_f1 = test(model, train_data, device)

    print("\nTest Set Evaluation:")
    test_acc, test_precision, test_recall, test_f1 = test(model, test_data, device)

    # 可视化训练损失
    plot_losses(train_losses)

    # 保存模型
    save_model(model)

    # 加载模型
    load_model(model, device=device)

    # 进行预测
    predictions = predict(model, test_data, device)
    print(f'Predictions: {predictions}')


if __name__ == "__main__":
    # 选择模型类型（'GCN', 'GraphSAGE', 'GAT', 'HAT'）
    # 例如使用 'GAT' 模型
    main(model_type='GAT')
