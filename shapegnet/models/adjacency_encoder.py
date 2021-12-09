import numpy as np


class AdjacencyEncoder:
    """

    """
    def __init__(self, max_prev_node=10, device='cpu'):
        self.device = device
        self.max_prev_node = max_prev_node

    def encode(self, adj, depth=10, is_full=False):
        """

        :param adj: n*n, rows means time step, while columns are input dimension
        :param depth: we want to keep row number, but truncate column numbers
        :param is_full:

        :return:
        """
        if is_full:
            self.max_prev_node = adj.shape[0] - 1

        # pick up lower tri
        adj = np.tril(adj, k=-1)
        n = adj.shape[0]
        adj = adj[1:n, 0:n - 1]

        # use max_prev_node to truncate
        # note: now adj is a (n-1) * (n-1) matrix
        adj_output = np.zeros((adj.shape[0], depth))
        for i in range(adj.shape[0]):
            input_start = max(0, i - depth + 1)
            input_end = i + 1
            output_start = depth + input_start - input_end
            output_end = depth
            adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
            # reverse order
            adj_output[i, :] = adj_output[i, :][::-1]

        return adj_output
