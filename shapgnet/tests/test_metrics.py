import networkx as nx
from shapgnet.utils import perturb


def test_perturbed():
    graphs = []
    for i in range(100, 101):
        for j in range(4, 5):
            for k in range(500):
                graphs.append(nx.barabasi_albert_graph(i, j))
    g_perturbed = perturb(graphs, 0.9)
    print([g.number_of_edges() for g in graphs])
    print([g.number_of_edges() for g in g_perturbed])
