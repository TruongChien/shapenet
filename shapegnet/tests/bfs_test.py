import networkx as nx
from shapegnet.models.sampler.utils import bfs_paths

g = nx.ladder_graph(4)
# get ordered at level bfs
paths = bfs_paths(g, 0)
print(paths)