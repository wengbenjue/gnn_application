import torch
import torch_geometric as pyg
from torch_geometric.data import Data, DataLoader

# 创建一个随机图数据集，每个图有10个节点，每个节点有3个特征，每个图有2个类别
num_graphs = 100  # 数据集中的图的数量
num_nodes = 10  # 每个图中的节点数量
num_features = 3  # 每个节点的特征数量
num_classes = 2  # 每个图的类别数量

# 随机生成节点特征矩阵，维度为(num_nodes, num_features)
x = torch.randn((num_nodes, num_features))

# 随机生成边索引矩阵，维度为(2, num_edges)，每一列表示一条边的两个端点
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

# 随机生成图标签向量，维度为(num_graphs,)
y = torch.randint(0, num_classes, (num_graphs,))

# 使用Data类将节点特征、边索引和图标签封装成一个数据对象
data = Data(x=x, edge_index=edge_index, y=y)

# 使用DataLoader类将数据对象转换成一个可迭代的数据加载器，指定batch_size和shuffle参数
loader = DataLoader(data, batch_size=32, shuffle=True)

# 切分训练集和测试集，比例为8:2
train_size = int(num_graphs * 0.8)
test_size = num_graphs - train_size
train_loader, test_loader = torch.utils.data.random_split(loader, [train_size, test_size])

# 打印训练集和测试集的大小
print(f"Train size: {len(train_loader)}")
print(f"Test size: {len(test_loader)}")


