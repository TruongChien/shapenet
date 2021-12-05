from queue import LifoQueue as stack
from queue import SimpleQueue as queue

import networkx as nx
import pylab as plt
from IPython.core.display import HTML, display

# import pygraphviz
# from networkx.drawing.nx_agraph import graphviz_layout


def bfs_seq(input_graph, start_id):
    """
     Get a bfs node sequence.

    :param input_graph:
    :param start_id:
    :return:
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


def gc(qe):
    if not qe.empty():
        while not qe.empty():
            qe.get()


def bdfs(graph : nx.classes.graph.Graph, start, goal, search='dfs'):
    """
    This is a template. Taking fringe = stack() gives DFS and
    fringe = queue() gives BFS. We need to add a priority function to get UCS.

    Usage: bp = bdfs(graph, start, goal, queue_or_stack = stack()) (this is dfs)
           bp = bdfs(graph, start, goal, queue_or_stack = queue()) (this is bfs)
    """
    depth = {}
    if search == 'dfs':
        queue_or_stack = stack()
        weight = -1
    else:
        queue_or_stack = queue()
        weight = 1

    gc(queue_or_stack)
    current = start
    closed = set()
    back_pointer = {}
    depth[start] = 0
    queue_or_stack.put(current)

    while True:
        if queue_or_stack.empty():
            return None
        while True:
            current = queue_or_stack.get()
            if current not in closed:
                break
            if queue_or_stack.empty():
                return None

        closed.add(current)
        if current == goal:
            return back_pointer

        if graph[current]:
            for node in graph[current]:
                if node not in closed:
                    node_depth = depth[current] + weight
                    if node not in depth or node_depth < depth[node]:
                        back_pointer[node] = current
                        depth[node] = node_depth
                    queue_or_stack.put(node)


def dfs(graph : nx.classes.graph.Graph, start, goal):
    return bdfs(graph, start, goal, search='dfs')


def bfs(graph : nx.classes.graph.Graph, start, goal):
    return bdfs(graph, start, goal, search='bfs')


def get_gr(digraph=True):
    if digraph:
        return nx.DiGraph()
    else:
        return nx.Graph()


def adj2graph(graph : nx.classes.graph.Graph, digraph=True):
    """
    for list representation of adj
    """
    gr = get_gr(digraph=digraph)
    for node in graph:
        gr.add_node(node)
        if graph[node]:
            for adj in graph[node]:
                gr.add_edge(node, adj)
                gr[node][adj]['weight'] = graph[node][adj]
    return gr


def edges_color(graph : nx.classes.graph.Graph, bfs_edge):
    """
    Return edge color and edge weight for a bfs path
    """
    edge_col = ['purple' if e in bfs_edge else 'blue' if e in bfs_edge else 'orange' for e in graph.edges()]
    edge_width = [3 if e in bfs_edge else 1 for e in graph.edges()]
    return edge_col, edge_width


def show_graph(graph : nx.classes.graph.Graph, start: int, goal: int, node_labels='default',
               node_pos='neato', plot_size=(14, 14), file_name=None, is_digraph=True):
    """
    node_labels label to use: 'default', 'none', or a list of labels to use.
    file_name -  a file nama 'my_graph.png'
    """
    fig, ax = plt.subplots(figsize=plot_size)
    Gr = g

    if node_pos == 'project_layout':
        node_pos = dict(zip(Gr.nodes(), [(b, 9 - a) for a, b in Gr.nodes()]))
    else:
        node_pos = nx.nx_pydot.graphviz_layout(Gr, prog=node_pos, root=start)

    edge_weight = nx.get_edge_attributes(Gr, 'weight')

    def path_edges(_path):
        """
        @param _path:
        @return:
        """
        edges = list(zip(_path[:-1], _path[1:]))
        # print(type(Gr[z[0]][z[1])
        # cost = sum([Gr[z[0]][z[1]]['weight'] for z in edges])
        if not is_digraph:
            edges += list(zip(_path[1:], _path[:-1]))
        return edges, 1

    bfs_path = getPath(bdfs(graph, start, goal, search='bfs'), start, goal)
    bfs_edge, bfs_cost = path_edges(bfs_path)

    node_col = ['red' if node in bfs_path else 'lightgray' for node in Gr.nodes()]

    if node_labels == 'default':
        nodes = nx.draw_networkx_nodes(Gr, node_pos, ax=ax, node_color=node_col, node_size=400)
        nodes.set_edgecolor('k')
        nx.draw_networkx_labels(Gr, node_pos, ax=ax, font_size=8)
    elif node_labels == 'none':
        nodes = nx.draw_networkx_nodes(Gr, node_pos, ax=ax, node_color=node_col, node_size=50)
    else:
        # labels must be a list
        nodes = nx.draw_networkx_nodes(Gr, node_pos, ax=ax, node_color=node_col, node_size=400)
        nodes.set_edgecolor('k')
        mapping = dict(zip(Gr.nodes, node_labels))
        nx.draw_networkx_labels(Gr, node_pos, labels=mapping, ax=ax, font_size=8)

    edge_col, edge_width = edges_color(Gr, bfs_edge)

    if is_digraph:
        nx.draw_networkx_edge_labels(Gr, node_pos, ax=ax, label_pos=0.3, edge_labels=edge_weight)
    else:
        nx.draw_networkx_edge_labels(Gr, node_pos, ax=ax, edge_labels=edge_weight)
    nx.draw_networkx_edges(Gr, node_pos, ax=ax, edge_color=edge_col, width=edge_width, alpha=.3)

    if file_name:
        plt.savefig(file_name)

    plt.show()
    display(HTML())


def getPath(bp, start: int, goal : int):
    """
    @param bp: back pointer
    @param start:
    @param goal:
    @return:
    """
    current = goal
    s = [current]
    while current != start:
        current = bp[current]
        s += [current]
    return list(reversed(s))


# create test graph
g = nx.ladder_graph(4)
# get ordered at level bfs
paths = bfs_seq(g, 0)
# display each step
for path in paths:
    show_graph(g, 0, path, plot_size=(6, 6))
# ladder_graph(n)
print(paths)
