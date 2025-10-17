import numpy as np
from graph_construction.build_global_graph import construct_global_graph
from graph_construction.visualize_graph import visualize_graph

features = np.load("features/caltech_features.npy")
image_paths = np.load("features/image_paths.npy")

graph_data = construct_global_graph(features, image_paths, k=5)
print(graph_data)

visualize_graph(graph_data, seed=42)
