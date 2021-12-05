import numpy as np

from .AbstractGraphDecoder import AbstractGraphDecoder


class AdjacencyDecoder(AbstractGraphDecoder):
    """

    """
    def __init__(self, device='cpu'):
        super().__init__(device)

    @staticmethod
    def decode(adj_output):
        """
            recover to adj from adj_output
            note: here adj_output have shape (n-1) * m
        """
        max_prev_node = adj_output.shape[1]

        # adj matrix
        adj = np.zeros((adj_output.shape[0],
                        adj_output.shape[0]))

        for i in range(adj_output.shape[0]):
            # start
            block_start = max(0, i - max_prev_node + 1)
            block_end = i + 1
            #
            output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
            output_end = max_prev_node
            # reverse order
            adj[i, block_start:block_end] = adj_output[i, ::-1][output_start:output_end]

        adj_full = np.zeros((adj_output.shape[0] + 1, adj_output.shape[0] + 1))
        n = adj_full.shape[0]

        adj_full[1:n, 0:n - 1] = np.tril(adj, 0)
        adj_full = adj_full + adj_full.T

        return adj_full
