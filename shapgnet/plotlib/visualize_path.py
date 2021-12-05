from sys import stderr
from queue import LifoQueue as stack
from queue import PriorityQueue as p_queue
from queue import SimpleQueue as queue
import networkx as nx
import pylab as plt
from IPython.core.display import HTML, display, Image


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

    # print("start", start_id)
    # print("output", len(output))

    return output


# import pygraphviz
# from networkx.drawing.nx_agraph import graphviz_layout

# It seems like these structures can keep "garbage" fro
# previous runs, so we must clean them out before using:
def gc(queue):
    if not queue.empty():
        while not queue.empty():
            queue.get()


ToyGraph = {0: {1: 1, 2: 1},
            1: {3: 8},
            2: {4: 2},
            3: {4: 1, 6: 2},
            4: {5: 2, 3: 5},
            5: {3: 1, 4: 2},
            6: {}}


def bdfs(G, start, goal, search='dfs'):
    """
    This is a template. Taking fringe = stack() gives DFS and
    fringe = queue() gives BFS. We need to add a priority function to get UCS.

    Usage: back_pointer = bdfs(G, start, goal, fringe = stack()) (this is dfs)
           back_pointer = bdfs(G, start, goal, fringe = queue()) (this is bfs)
    """

    # There is actually a second subtle difference between stack and queue and that
    # has to do with when one revises the pack_pointer. Essentially, this amounts to
    # defining a priority function where queue prioritizes short paths, fat search trees
    # while dfs prioritizes long paths, skinny search trees.
    depth = {}

    if search == 'dfs':
        fringe = stack()
        weight = -1  # We are pretending all edges have weight -1
    else:
        fringe = queue()
        weight = 1  # We are pretending all edges have weight 1

    gc(fringe)  # Make sure there is no garbage in the fringe

    closed = set()
    back_pointer = {}
    current = start
    depth[start] = 0
    fringe.put(current)

    while True:
        # If the fringe becomes empty we are out of luck
        if fringe.empty():
            print("There is no path from {} to {}".format(start, goal), file=stderr)
            return None

        # Get the next closed element of the closed set. This is complicated
        # by the fact that our queue has no delete so items that are already
        # in the closed set might still be in the queue. We must make sure not
        # to choose such an item.
        while True:
            current = fringe.get()
            if current not in closed:
                break
            if fringe.empty():
                print("There is no path from {} to {}".format(start, goal), file=stderr)
                return None

        # Add current to the closed set
        closed.add(current)

        # If current is the goal we are done.
        if current == goal:
            return back_pointer

        # Add nodes adjacent adjacent to current to the fringe
        # provided they are not in the closed set.
        if G[current]:  # Check if G[current] != {}, bool({}) = False
            for node in G[current]:
                if node not in closed:
                    node_depth = depth[current] + weight
                    if node not in depth or node_depth < depth[node]:
                        back_pointer[node] = current
                        depth[node] = node_depth
                    fringe.put(node)


def dfs(G, start, goal):
    return bdfs(G, start, goal, search='dfs')


def bfs(G, start, goal):
    return bdfs(G, start, goal, search='bfs')


def adjToNxGraph(G, digraph=True):
    """
    Converts one of our adjacency "list" representations for a graph into
    a networkx graph.
    """
    if digraph:
        Gr = nx.DiGraph()
    else:
        Gr = nx.Graph()

    for node in G:
        Gr.add_node(node)
        if G[node]:
            for adj in G[node]:
                Gr.add_edge(node, adj)
                Gr[node][adj]['weight'] = G[node][adj]
    return Gr


def showGraph(G, start, goal, paths=[], node_labels='default',
              node_pos='neato', gsize=(14, 14), save_file=None, digraph=True):
    """
    paths should be an array of which paths to show: paths = ['bfs', 'dfs', 'ucs']
    node_labels must be one of: 'default', 'none', or a list of labels to use.
    save_file must be an image file name with extension, i.e., save_file='my_graph.png'
    """

    fig, ax = plt.subplots(figsize=gsize)

    # Convert G into structure used in networkx
    # Gr = adjToNxGraph(G, digraph=digraph)
    Gr = g

    if node_pos == 'project_layout':
        # The project graphs have a particular structure.
        node_pos = dict(zip(Gr.nodes(), [(b, 9 - a) for a, b in Gr.nodes()]))
    else:
        node_pos = nx.nx_pydot.graphviz_layout(Gr, prog=node_pos, root=start)

    edge_weight = nx.get_edge_attributes(Gr, 'weight')

    def path_edges(path):
        """

        @param path:
        @return:
        """
        edges = list(zip(path[:-1], path[1:]))

        #print(type(Gr[z[0]][z[1])

        #cost = sum([Gr[z[0]][z[1]]['weight'] for z in edges])

        if not digraph:
            edges += list(zip(path[1:], path[:-1]))

        return edges, 1

    # Process Paths:
    if 'bfs' in paths:
        bpath = getPath(bdfs(G, start, goal, search='bfs'), start, goal)
        bedges, bcost = path_edges(bpath)
    else:
        bpath = []
        bedges = []

    if 'dfs' in paths:
        dpath = getPath(bdfs(G, start, goal, search='dfs'), start, goal)
        dedges, dcost = path_edges(dpath)
    else:
        dpath = []
        dedges = []

    if 'ucs' in paths:
        ucost, back = ucs(G, start, goal)
        upath = getPath(back, start, goal)
        uedges, ucost = path_edges(upath)
    else:
        upath = []
        uedges = []

    node_col = ['orange' if node in upath
                else 'purple' if node in bpath and node in dpath
    else 'blue' if node in dpath
    else 'red' if node in bpath
    else 'lightgray' for node in Gr.nodes()]

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

    edge_col = ['purple' if edge in bedges and edge in dedges
                else 'blue' if edge in dedges
    else 'red' if edge in bedges
    else 'orange' if edge in uedges else 'gray' for edge in Gr.edges()]

    edge_width = [3 if edge in dedges or edge in bedges or edge in uedges else 1 for edge in Gr.edges()]

    if digraph:
        nx.draw_networkx_edge_labels(Gr, node_pos, ax=ax, label_pos=0.3, edge_labels=edge_weight)
        # nx.draw_networkx_edge_labels(Gr, node_pos, ax=ax, edge_color=edge_col, label_pos=0.3, edge_labels=edge_weight)
    else:
        nx.draw_networkx_edge_labels(Gr, node_pos, ax=ax, edge_labels=edge_weight)
    nx.draw_networkx_edges(Gr, node_pos, ax=ax, edge_color=edge_col, width=edge_width, alpha=.3)

    if save_file:
        plt.savefig(save_file)

    plt.show()

    result = "DFS gives a path of length {} with cost {}<br>".format(len(dpath) - 1, dcost) if 'dfs' in paths else ""
    result += "BFS gives a path of length {} with cost {}. BFS always returns a minimal length path.<br>".format(
        len(bpath) - 1, bcost) if 'bfs' in paths else ""
    result += "UCS gives a path of length {} with cost {}. UCS always returns a minimal cost path.".format(
        len(upath) - 1, ucost) if 'ucs' in paths else ""

    display(HTML(result))  # Need display in Jupyter


def getPath(backPointers, start, goal):
    """

    @param backPointers:
    @param start:
    @param goal:
    @return:
    """
    current = goal
    s = [current]
    while current != start:
        current = backPointers[current]
        s += [current]

    return list(reversed(s))


def ucs(G, start, goal, trace=False):
    """

    This returns the least cost of a path from start to goal or reports
    the non-existence of such path.

    This also returns a pack_pointer from
    which the search tree can be reconstructed as well as all paths explored
    including the one of interest.

    @param G:
    @param start:
    @param goal:
    @param trace:
    @return:
    """
    """
    

    Usage: cost, back_pointer = ucs(Graph, start, goal)
    """

    # Make sure th queue is empty. (Bug in implementation?)
    fringe = p_queue()
    gc(fringe)

    # If we did not care about the path, only the cost we could
    # omit this block.
    cost = {}  # If all we want to do is solve the optimization
    back_pointer = {}  # problem, neither of these are necessary.
    cost[start] = 0
    # End back_pointer/cost block

    current = start
    fringe.put((0, start))  # Cost of start node is 0
    closed = set()

    while True:
        # If the fringe becomes empty we are out of luck
        if fringe.empty():
            print("There is no path from {} to {}".format(start, goal), file=stderr)
            return None

        # Get the next closed element of the closed set. This is complicated
        # by the fact that our queue has no delete so items that are already
        # in the closed set might still be in the queue. We must make sure not
        # to choose such an item.
        while True:
            current_cost, current = fringe.get()
            if current not in closed:
                # Add current to the closed set
                closed.add(current)
                if trace:
                    print("Add {} to the closed set with cost {}".format(current, current_cost))
                break
            if fringe.empty():
                print("There is no path from {} to {}".format(start, goal), file=stderr)
                return None

        # If current is the goal we are done.
        if current == goal:
            return current_cost, back_pointer

        # Add nodes adjacent to current to the fringe
        # provided they are not in the closed set.
        if G[current]:  # Check if G[current] != {}, bool({}) = False
            for node in G[current]:
                if node not in closed:
                    node_cost = current_cost + G[current][node]

                    # Note this little block could be removed if we only
                    # cared about the final cost and not the path
                    if node not in cost or cost[node] > node_cost:
                        back_pointer[node] = current
                        cost[node] = node_cost
                        if trace:
                            print("{current} <- {node}".format(current, node))
                    # End of back/cost block.

                    fringe.put((node_cost, node))
                    if trace:
                        print("Add {} to fringe with cost {}".format(node, node_cost))


# show bfs path
# showGraph(ToyGraph, 0, 6, paths=['bfs'], gsize=(8, 8))
# ucs

# G = nx.grid_2d_graph(4, 4)  # 4x4 grid
# G = nx.barbell_graph(4, 4)
# showGraph(G, 0, 4, paths=['bfs'], gsize=(4, 4))

# print(nx.bfs_successors(ToyGraph, 1))
# paths = bfs_seq(nx.barbell_graph(4, 4), 3)
g = nx.ladder_graph(4)
paths = bfs_seq(g, 0)

for path in paths:
    showGraph(g, 0, path, paths=['bfs'], gsize=(4, 4))

# ladder_graph(n)
print(paths)

# showGraph(ToyGraph, 0, 6, paths=['bfs', 'ucs'], gsize=(8, 8))
