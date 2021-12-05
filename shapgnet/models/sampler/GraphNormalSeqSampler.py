import numpy as np
import networkx as nx
import torch

# use pytorch dataloader
from shapegnet.models.adjacency_encoder import AdjacencyEncoder


class GraphSeqNormalSampler(torch.utils.data.Dataset):
    def __init__(self, graphs, max_num_node=None):
        self.adj_all = []
        self.len_all = []
        for G in graphs:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())

        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node

        self.encoder = AdjacencyEncoder()

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        # here zeros are padded for small graph
        x_batch = np.zeros((self.n, self.n - 1))
        # the first input token is all ones
        x_batch[0, :] = 1
        # here zeros are padded for small graph
        y_batch = np.zeros((self.n, self.n - 1))
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = self.encoder.encode(adj_copy.copy(), max_prev_node=self.n - 1)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x': x_batch, 'y': y_batch, 'len': len_batch}

# dataset = GraphSeqNormalSampler(graphs)
# print(dataset[1]['x'])
# print(dataset[1]['y'])
# print(dataset[1]['len'])
