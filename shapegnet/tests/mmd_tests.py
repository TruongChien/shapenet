import networkx as nx
import numpy as np

from shapgnet.models.adjacency_encoder import AdjacencyEncoder
from shapgnet.models.adjacency_decoder import AdjacencyDecoder
from shapgnet.external.graphrnn_eval.stats import degree_stats, clustering_stats, orbit_stats_all
import matplotlib.pyplot as plt
import time
from datetime import timedelta

np.random.seed(1234)


def test_decoder(plot=False):
    original_graph = nx.erdos_renyi_graph(10, 0.4)
    original_graph_as_np = nx.to_numpy_array(original_graph)
    print(original_graph_as_np)
    if plot:
        nx.draw(original_graph, with_labels=True)
        plt.show()

    print("Adj matrix for random graph")
    print(original_graph._adj)
    encoder = AdjacencyEncoder()
    encoded_adj = encoder.encode(original_graph_as_np.copy(), 10)

    print("Encoded matrix")
    print(encoded_adj)
    print("---------------------------")

    decoder = AdjacencyDecoder()
    decoded = decoder.decode(encoded_adj)
    print("Decoder matrix")
    print(decoded)
    print(type(decoded))
    print(decoded.shape)
    print("---------------------------")

    decoded_graph = nx.from_numpy_array(decoded.copy())
    print("Graph type", type(decoded_graph))
    print(decoded_graph.edges(data=True))
    if plot:
        nx.draw(decoded_graph, with_labels=True)
        plt.show()

    print("Original graph", type(original_graph_as_np))
    # Now let compute metric between two graph in simular way as it describe in GraphRNN
    # and NetGAN paper

    print("Running mmd degree")
    mmd_degree = degree_stats([original_graph], [decoded_graph])
    print(mmd_degree)

    print("Running cluster stats")
    mmd_clustering = clustering_stats([original_graph], [decoded_graph])
    print(mmd_clustering)

    print("Running orbits")
    start_time = time.monotonic()
    mmd_4orbits = orbit_stats_all([original_graph], [decoded_graph])
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    print(mmd_4orbits)


def test_large_graph(num_graph=10, num_nodes=10):
    """
    Test for mmd stats.
    """
    original_graphs = []
    for n in range(1, num_graph):
        original_graphs.append(nx.erdos_renyi_graph(num_nodes, 0.4))

    decoded_graphs = []
    for n in range(1, 100):
        decoded_graphs.append(nx.erdos_renyi_graph(num_nodes, 0.4))

    start_time = time.monotonic()
    mmd_degree = degree_stats(original_graphs, decoded_graphs)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    #print(mmd_degree)

    print("Running cluster stats")
    start_time = time.monotonic()
    mmd_clustering = clustering_stats(original_graphs, decoded_graphs)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    #print(mmd_clustering)

    #print("Running orbits")
    start_time = time.monotonic()
    mmd_4orbits = orbit_stats_all(original_graphs, decoded_graphs)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    #print(mmd_4orbits)


if __name__ == '__main__':
    # test_decoder()
    test_large_graph()
