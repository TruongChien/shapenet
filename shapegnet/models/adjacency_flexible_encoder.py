import numpy as np


class AdjacencyFlexEncoder:
    """

    """
    def __init__(self, depth=10, device='cuda'):
        self.device = device
        self.max_prev_node = depth

    @staticmethod
    def encode(adj):
        """

        :return:
        """
        adj = np.tril(adj, k=-1)
        n = adj.shape[0]
        adj = adj[1:n, 0:n - 1]

        adj_output = []
        input_start = 0
        for i in range(adj.shape[0]):
            input_end = i + 1
            adj_slice = adj[i, input_start:input_end]
            adj_output.append(adj_slice)
            non_zero = np.nonzero(adj_slice)[0]
            input_start = input_end - len(adj_slice) + np.amin(non_zero)

        return adj_output
