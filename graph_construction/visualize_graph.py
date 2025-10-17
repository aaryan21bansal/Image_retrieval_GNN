import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_graph(graph_data, num_nodes=None, seed: int = 42):
    G = to_networkx(graph_data, to_undirected=True)

    if num_nodes is not None and num_nodes < graph_data.num_nodes:
        nodes = list(range(num_nodes))
        G = G.subgraph(nodes)

    pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(12, 12))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=50,
        edge_color="gray",
        node_color="blue",
        alpha=0.7
    )
    plt.show()
