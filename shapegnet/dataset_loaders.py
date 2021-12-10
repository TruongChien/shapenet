# ShapeGen
#
# A dataset loader and utilities function.
#
#
# Author Mustafa Bayramov
import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


# def parse_index_file(filename):
#     [int(l.strip()) for l in open(filename)]
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index

def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    """
    Parse point cloud
    """
    """
    Parse point cloud
    """
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    """
    Read from text file to array and create tensor.
    Point cloud stored in array x y z and normal vectors
    """
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def parse_index_file(filename):
    return [int(i.strip()) for i in open(filename)]


def caveman_special(c=2, k=20, p_path=0.1, p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)), 1)

    graph = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1 - p_edge
    for (u, v) in list(graph.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            graph.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        graph.add_edge(u, v)

    subgraph = (graph.subgraph(c) for c in nx.connected_components(graph))
    subgraph = max(subgraph, key=len)
    return subgraph


def graph_dataset(dataset='cora', enc='latin1'):
    """
    Load cora and pubmed dataset
    :param dataset: dataset name
    :param enc type
    :return:
    """
    names = ['x', 'tx', 'allx', 'graph']
    objects = [pkl.load(open("dataset/ind.{}.{}".format(dataset, n), 'rb'),
                        encoding=enc) for n in range(len(names))]

    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    # features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    #
    nx_graph = nx.from_dict_of_lists(graph)
    return nx.adjacency_matrix(nx_graph), features, nx_graph
