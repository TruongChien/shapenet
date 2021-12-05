import matplotlib.pyplot as plt
import networkx as nx
from networkx import ladder_graph, barbell_graph, lollipop_graph

# G = nx.grid_2d_graph(5, 5)  # 5x5 grid
# G = ladder_graph(12, create_using=None)
# G = barbell_graph(6, 6)
G = lollipop_graph(6, 6)

graphs = []
for i in range(2, 5):
    for j in range(2, 6):
        graphs.append(nx.grid_2d_graph(i, j))

# print the adjacency list
for line in nx.generate_adjlist(G):
    print(line)
# write edgelist to grid.edgelist
nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
# read edgelist from grid.edgelist
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

pos = nx.spring_layout(H, seed=200)
nx.draw(H, pos)
plt.show()