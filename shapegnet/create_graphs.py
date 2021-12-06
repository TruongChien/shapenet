import sys
from random import shuffle

import networkx as nx
from .dataset_loaders import graph_dataset, caveman_special
from .model_config import ModelSpecs
from .utils import fmt_print


def dataset_graph_generator():
    """ Dataset create"""
    return {
        'grid':          generate_grid,
        'grid_small':    generate_grid,
        'grid_big':      generate_grid,
        'grid_min':      generate_grid,
        'caveman':       generate_caveman,
        'caveman_small': generate_caveman,
        'caveman_big':   generate_caveman,
    }


def gracefully_exit(msg):
    """

    """
    print(msg)
    sys.exit()


def generate_grid(specs: ModelSpecs):
    """
    Generated grid based on spec.
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.lattice.grid_2d_graph.html

    @param specs:
    @return:
    """
    if specs.is_graph_creator_verbose():
        fmt_print("Generating grid graph type", "backend nx")

    graph_spec = specs.graph_specs['graph_spec']

    if 'grid_n' not in graph_spec:
        gracefully_exit("Grid graph must contain grid_n")
        sys.exit()

    if 'grid_m' not in graph_spec:
        gracefully_exit("Grid graph must contain grid_m")
        sys.exit()

    grid_n = graph_spec['grid_n']
    grid_m = graph_spec['grid_m']

    if specs.is_graph_creator_verbose():
        fmt_print("Synthetic grid's n and m", grid_n, grid_m)

    graphs = []
    for i in range(grid_n[0], grid_n[1]):
        for j in range(grid_m[0], grid_m[1]):
            graphs.append(nx.grid_2d_graph(i, j))

    return graphs


def citeseer(radius=1, min_nodes=4, max_nodes=20, split=200, is_shuffled=True):
    """

    """
    _, _, g = graph_dataset(dataset='citeseer')
    G = max(nx.connected_component_subgraphs(g), key=len)
    G = nx.convert_node_labels_to_integers(G)

    graphs = []
    for i in range(G.number_of_nodes()):
        ego_graph = nx.ego_graph(G, i, radius=radius)
        num_nodes = ego_graph.number_of_nodes()
        if min_nodes <= num_nodes <= max_nodes:
            graphs.append(ego_graph)

    if is_shuffled:
        shuffle(graphs)

    graphs = graphs[0:split]
    return 15, graphs


def ladder_graph(start_range, stop_range):
    """

    """
    print("Creating ladder graph..")
    graphs = []
    for i in range(start_range, stop_range):
        graphs.append(nx.ladder_graph(i))
    return 10, graphs


def tree_graph(ix, iy, jx, jy, is_verbose):
    """

    """
    if is_verbose:
        print("Creating tree graph {} {} {} {}".format(ix, iy, jx, jy))

    g = []
    for i in range(ix, iy):
        for j in range(jx, jy):
            g.append(nx.balanced_tree(i, j))

    return 256


def generate_caveman(specs: ModelSpecs):
    """
    Parameters
      l int number of cliques
      k int size of cliques (k at least 2 or NetworkXError is raised)
    """
    if specs.is_graph_creator_verbose():
        fmt_print("Generating graph type", "caveman nx graph")

    graph_spec = specs.graph_specs['graph_spec']
    num_of_cliques_i = graph_spec['num_of_cliques_i']
    num_of_cliques_j = graph_spec['num_of_cliques_j']
    k = graph_spec['size_of_cliques']
    p = graph_spec['p_edge']

    if specs.is_graph_creator_verbose():
        fmt_print("Number of number of cliques", num_of_cliques_i, num_of_cliques_j)
        fmt_print("Size of cliques k", k)
        fmt_print("Probability of edges", p)

    graphs = []
    for i in range(num_of_cliques_i[0], num_of_cliques_i[1]):
        for j in range(num_of_cliques_j[0], num_of_cliques_j[1]):
            for k in range(k):
                graphs.append(caveman_special(i, j, p_edge=p))

    return graphs


def create(specs: ModelSpecs):
    """
    Method lookup graph and forward to respected creator.
    """
    dispatch = dataset_graph_generator()
    g = dispatch[specs.active](specs)
    return g