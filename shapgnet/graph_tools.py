# load a list of graphs
import pickle
import networkx as nx


def connected_component(g, n=0):
    """
    First get set of nodes in the component of graph containing node n.
    return a subgraph belonging to a given cluster.
    """
    node_list = nx.node_connected_component(g, n)
    return g.subgraph(node_list)


def pick_connected_component_new(g):
    """

    """
    adj_list = g.adjacency_list()
    for i, adj in enumerate(adj_list):
        if min(adj) > i >= 1:
            break

    g = g.subgraph(list(range(i)))
    subgraph = [c for c in nx.connected_components(g)]
    g = max(subgraph, key=len)

    return g


def graph_from_tensors(g, is_real=True):
    """

    """
    loop_edges = list(nx.selfloop_edges(g))
    if len(loop_edges) > 0:
        g.remove_edges_from(loop_edges)
    if is_real:
        subgraph = (g.subgraph(c) for c in nx.connected_components(g))
        g = max(subgraph, key=len)
        g = nx.convert_node_labels_to_integers(g)
    else:
        g = pick_connected_component_new(g)

    return g


def graph_from_file(file_name, is_real=True):
    """
    Loads existing graph file from a file.
    """
    with open(file_name, "rb") as f:
        graph_list = pickle.load(f)
        print("Loading {}, file contains {} graphs".format(file_name, len(graph_list)))

    for i in range(len(graph_list)):
        loop_edges = list(nx.selfloop_edges(graph_list[i]))
        if len(loop_edges) > 0:
            graph_list[i].remove_edges_from(loop_edges)
        if is_real:
            subgraph = (graph_list[i].subgraph(c) for c in nx.connected_components(graph_list[i]))
            graph_list[i] = max(subgraph, key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])

    return graph_list


def export_graphs_to_txt(g_list, output_filename_prefix):
    i = 0
    for G in g_list:
        f = open(output_filename_prefix + '_' + str(i) + '.txt', 'w+')
        for (u, v) in G.edges():
            idx_u = G.nodes().index(u)
            idx_v = G.nodes().index(v)
            f.write(str(idx_u) + '\t' + str(idx_v) + '\n')
        i += 1
