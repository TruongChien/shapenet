import networkx as nx
import numpy as np


def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = np.sum(A, axis=1) + 1

    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(np.power(degrees, -0.5).flatten())
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    A_normal = np.dot(np.dot(D, A_hat), D)
    return A_normal


def bfs_paths(input_graph, start_id):
    """
    Get a bfs node id sequence from some start_id

    :param input_graph: networkx graph
    :param start_id: node id
    :return: BFS order.
    """
    dictionary = dict(nx.bfs_successors(input_graph, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        frontier = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                frontier = frontier + neighbor
        output = output + frontier
        start = frontier

    return output
