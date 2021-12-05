import numpy as np
import networkx as nx
import torch

# use pytorch dataloader
from ..adjacency_encoder import AdjacencyEncoder
from ..adjacency_flexible_encoder import AdjacencyFlexEncoder
from ...models.sampler.utils import bfs_seq


class GraphSeqSampler(torch.utils.data.Dataset):
    """

    """

    def __init__(self, graphs, max_num_node=None, max_prev_node=None, max_iteration=20000):

        self.sorted = False
        self.adj_all = []
        self.len_all = []

        for g in graphs:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(g)))
            self.len_all.append(g.number_of_nodes())

        self.n = max_num_node
        self.max_prev_node = max_prev_node

        if max_num_node is None:
            self.n = max(self.len_all)

        if max_prev_node is None:
            self.max_prev_node = max(self.calc_max_prev_node(iter=max_iteration))

        if self.sorted:
            self.max_prev_node = max_prev_node
            # sort Graph in descending order
            len_batch_order = np.argsort(np.array(self.len_all))[::-1]
            self.len_all = [self.len_all[i] for i in len_batch_order]
            self.adj_all = [self.adj_all[i] for i in len_batch_order]

        self.encoder = AdjacencyEncoder()
        self.flex_encoder = AdjacencyFlexEncoder(max_prev_node=self.max_prev_node)

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):

        #start_time = time.time()
        #
        adj_copy = self.adj_all[idx].copy()

        # here zeros are padded for small graph, the first input
        # token is all ones
        x_batch = np.zeros((self.n, self.max_prev_node))
        x_batch[0, :] = 1

        y_batch = np.zeros((self.n, self.max_prev_node))

        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])

        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)

        # then do bfs in the permuted G
        g = nx.from_numpy_matrix(adj_copy_matrix)
        x_idx = np.array(bfs_seq(g,  np.random.randint(adj_copy.shape[0])))

        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = self.encoder.encode(adj_copy.copy(), max_prev_node=self.max_prev_node)

        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded

        #print("--- %s seconds ---" % (time.time() - start_time))
        return {'x': x_batch, 'y': y_batch, 'len': len_batch}

    def calc_max_prev_node(self, iter=20000, topk=10):
        """

        @param iter:
        @param topk:
        @return:
        """
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))

            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()

            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            g = nx.from_numpy_matrix(adj_copy_matrix)

            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(g, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]

            # encode adj
            adj_encoded = AdjacencyFlexEncoder.encode(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)

        max_prev_node = sorted(max_prev_node)[-1 * topk:]

        return max_prev_node
