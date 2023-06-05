import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, sort_edge_index
from torch_scatter import scatter_add

class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.6):
        super(GraphAttentionLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(heads, in_channels, out_channels))
        self.attention = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index_i, size_i):
        x_j = F.dropout(x_j, p=self.dropout, training=self.training)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.attention).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat:
            return aggr_out.view(-1, self.heads * self.out_channels)
        else:
            return aggr_out.mean(dim=1)

def softmax(src, index, num_nodes=None):
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out

class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers):
        super(GATClassifier, self).__init__()

        self.num_layers = num_layers

        self.embedding = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        self.layers.append(GraphAttentionLayer(hidden_channels, hidden_channels, num_heads))

        for _ in range(num_layers - 2):
            self.layers.append(GraphAttentionLayer(num_heads * hidden_channels, hidden_channels, num_heads))

        self.layers.append(GraphAttentionLayer(num_heads * hidden_channels, out_channels, 1, concat=False))

    def forward(self, x, edge_index):
        x = self.embedding(x)

        for i, layer in enumerate(self.layers):
            x = F.elu(layer(x, edge_index))

        return x

# 构建KNN图
def build_knn_graph(data, k, threshold):
    num_nodes = data.num_nodes
    batch_size = data.batch.max().item() + 1

    data_list = []
    for batch_id in range(batch_size):
        mask = (data.batch == batch_id)
        x = data.x[mask]
        edge_index = data.edge_index[:, mask]

        edge_index, _ = sort_edge_index(edge_index)
        edge_index = edge_index[:, :k]

        similarity = compute_similarity(x)
        mask = (similarity > threshold).nonzero(as_tuple=False).T
        edge_index = torch.cat([edge_index, mask], dim=1)

        data_list.append(Data(x=x, edge_index=edge_index))

    return Batch.from_data_list(data_list)

def compute_similarity(x):
    x_norm = torch.norm(x, dim=1, keepdim=True)
    x_normalized = x / (x_norm + 1e-8)
    similarity = torch.matmul(x_normalized, x_normalized.t())
    return similarity

# 模拟客户数据
num_nodes = 50000
num_features = 10
num_classes = 2

data = Data(x=torch.randn(num_nodes, num_features),
            y=torch.randint(0, num_classes, (num_nodes,)),
            edge_index=None)

# 将数据分成小批次进行处理
batch_size = 128
num_batches = (num_nodes + batch_size - 1) // batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GATClassifier(num_features, 64, num_classes, 8, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for batch_id in range(num_batches):
    start_idx = batch_id * batch_size
    end_idx = min((batch_id + 1) * batch_size, num_nodes)

    batch_data = data[start_idx:end_idx].to(device)

    batch_data.edge_index = None
    batch_graph = build_knn_graph(batch_data, k=5, threshold=0.5).to(device)
    batch_data.edge_index = batch_graph.edge_index

    optimizer.zero_grad()
    output = model(batch_data.x, batch_data.edge_index)
    loss = criterion(output, batch_data.y)
    loss.backward()
    optimizer.step()

    print(f'Batch {batch_id + 1}/{num_batches}, Loss: {loss.item()}')
