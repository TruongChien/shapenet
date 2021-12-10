import numpy as np


class AdjacencyFlexDecoder:
    def __init__(self, device='cuda'):
        self.device = device

    @staticmethod
    def decode(adj_output):
        """
            recover to adj from adj_output
            note: here adj_output have shape (n-1) * m
        """
        adj = np.zeros((len(adj_output), len(adj_output)))
        for i in range(len(adj_output)):
            output_start = i + 1 - len(adj_output[i])
            output_end = i + 1
            adj[i, output_start:output_end] = adj_output[i]
        adj_full = np.zeros((len(adj_output) + 1, len(adj_output) + 1))
        n = adj_full.shape[0]
        adj_full[1:n, 0:n - 1] = np.tril(adj, 0)
        adj_full = adj_full + adj_full.T

        return adj_full
