import networkx as nx
import numpy as np

from shapnet.models.adjacency_decoder import AdjacencyDecoder
from shapnet.models.adjacency_encoder import AdjacencyEncoder
from shapnet.models.sampler.utils import bfs_seq


def test_encode_decode_adj():
    """

    """

    encoder = AdjacencyEncoder(max_prev_node=5)
    decoder = AdjacencyDecoder()

    G = nx.ladder_graph(5)
    G = nx.grid_2d_graph(20, 20)
    G = nx.ladder_graph(200)
    G = nx.karate_club_graph()
    G = nx.connected_caveman_graph(2, 3)
    print(G.number_of_nodes())

    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    #
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]

    print('adj\n', adj)
    adj_output = encoder.encode(adj, max_prev_node=5)
    print('adj_output\n', adj_output)
    adj_recover = decoder.decode(adj_output)
    print('adj_recover\n', adj_recover)
    print('error\n', np.amin(adj_recover - adj), np.amax(adj_recover - adj))

    # adj_output = encode_adj_flexible(adj)
    # for i in range(len(adj_output)):
    #     print(len(adj_output[i]))
    # adj_recover = decode_adj_flexible(adj_output)
    # print(adj_recover)
    # print(np.amin(adj_recover - adj), np.amax(adj_recover - adj))


test_encode_decode_adj()