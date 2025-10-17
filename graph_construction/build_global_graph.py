import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np

def construct_global_graph(features, image_paths, k=5):
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float)

    num_images = features.size(0)
    features = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.matmul(features, features.t())
    edge_index = []
    for i in range(num_images):
        topk = torch.topk(sim_matrix[i], k+1).indices[1:]
        for j in topk:
            edge_index.append([i, j.item()])

    edge_index = torch.tensor(edge_index).t().contiguous()
    data = Data(x=features, edge_index=edge_index)
    data.image_paths = image_paths  
    return data
