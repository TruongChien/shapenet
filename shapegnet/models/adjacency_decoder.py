import numpy as np

from .AbstractGraphDecoder import AbstractGraphDecoder


class AdjacencyDecoder(AbstractGraphDecoder):
    """

    """
    def __init__(self, device='cpu'):
        super().__init__(device)

    @staticmethod
    def decode(encoder_output):
        """
        Decode back Adjacency Matrix from encoder output.

        @param encoder_output:
        @return:
        """
        max_prev_node = encoder_output.shape[1]
        adj = np.zeros((encoder_output.shape[0],
                        encoder_output.shape[0]))

        for i in range(encoder_output.shape[0]):
            block_start = max(0, i - max_prev_node + 1)
            block_end = i + 1
            output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
            output_end = max_prev_node
            adj[i, block_start:block_end] = encoder_output[i, ::-1][output_start:output_end]

        adj_full = np.zeros((encoder_output.shape[0] + 1, encoder_output.shape[0] + 1))
        n = adj_full.shape[0]

        adj_full[1:n, 0:n - 1] = np.tril(adj, 0)
        adj_full = adj_full + adj_full.T

        return adj_full
