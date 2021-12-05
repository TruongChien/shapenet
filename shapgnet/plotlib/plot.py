import io

import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torch

# draw a list of graphs [G]
from matplotlib import cm

from ..graph_tools import graph_from_file


def visualize_mesh(pos, face):
    """
    vizualize mesh
    :param pos:
    :param face:
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=data.face.t(), antialiased=False)
    plt.show()


def visualize_points(pos, edge_index=None, index=None):
    """
    Visualize cloud

    :param pos:
    :param edge_index:
    :param index:
    :return:
    """
    # fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
        mask = torch.zeros(pos.size(0), dtype=torch.bool)
        mask[index] = True
        plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
        plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()


def graph2image_buffer(graph, layout='spring', k=1, node_size=55,
                       alpha=1, width=1.3):
    plt.switch_backend('agg')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.axis("off")

    if layout == 'spring':
        pos = nx.spring_layout(graph, k=k / np.sqrt(graph.number_of_nodes()), iterations=100)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size, node_color='#336699', alpha=1, linewidths=0)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=alpha, width=width)

    with io.BytesIO() as buff:
        fig.savefig(buff, format='rgba')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)

    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))

    return im


def draw_samples_from_file(file_name, plot_type='train', file_prefix=None, num_samples=10):
    """

    @param file_name:
    @param plot_type:
    @param file_prefix:
    @param num_samples:
    @return:
    """
    if file_prefix is None:
        raise Exception("empty file prefix. ")

    path2file = str(Path(file_name))
    _graphs = graph_from_file(path2file)
    for i in range(0, len(_graphs)):
        if num_samples is not None and i == num_samples:
            break
        #
        draw_single_graph(_graphs[i], from_epoch=0,
                          graph_id=i,
                          plot_type=plot_type,
                          graph_name=plot_type,
                          file_prefix=file_prefix)


def draw_single_graph(g, file_name, graph_type='default', plot_type='prediction', graph_name='prediction',
                      layout='spring', k=1, node_size=100, alpha=1, width=1.3, backend='agg', dpi=600):
    """

    @param g:  networkx graph object
    @param file_name:
    @param layout:
    @param k:
    @param node_size:
    @param alpha:
    @param width:
    @return:
    :param graph_type:
    """
    """
    For community plot use louvain algo
        #    https:/python-louvain.readthedocs.io/en/latest/api.html

    """
    if file_name is None:
        raise Exception("Filename is none. You need provide file prefix for a plot.")

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    cmap = None
    if graph_type == 'community':
        # parts = community.best_partition(g)
        partion = community_louvain.best_partition(g)
        cmap = cm.get_cmap('viridis', max(partion.values()) + 1)

    plt.axis("off")
    if layout == 'spring':
        pos = nx.spring_layout(g, k=k / np.sqrt(g.number_of_nodes()), iterations=100)
        # pos = nx.spring_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(g)

    # nx.draw_networkx(G, with_labels=True, node_size=2, width=0.15, font_size = 1.5, node_color=colors,pos=pos)
    # nx.draw_networkx(G, with_labels=False, node_size=1.5, width=0.2, font_size = 1.5, linewidths=0.2, node_color = 'k',pos=pos,alpha=0.2)

    nx.draw_networkx_nodes(g, pos, cmap=cmap, node_size=node_size, node_color='#336699', alpha=1, linewidths=0)
    nx.draw_networkx_edges(g, pos, alpha=alpha, width=width)

    plt.axis('off')
    # plt.title('Complete Graph of Odd-degree Nodes')
    # plt.show()

    plt.tight_layout()

    print("Saving ", file_name)
    plt.savefig(file_name, dpi=600)
    plt.close()


def graph_to_image(g, graph_type='default', layout='spring',
                   k=1, node_size=100, alpha=1, width=1.3,
                   backend='agg',
                   dpi=600):
    """
    @param g:  networkx graph object
    @param file_name:
    @param layout:
    @param k:
    @param node_size:
    @param alpha:
    @param width:
    @return:
    """
    """

    """
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    cmap = None
    if graph_type == 'community':
        # parts = community.best_partition(g)
        partion = community_louvain.best_partition(g)
        cmap = cm.get_cmap('viridis', max(partion.values()) + 1)

    plt.axis("off")
    if layout == 'spring':
        pos = nx.spring_layout(g, k=k / np.sqrt(g.number_of_nodes()), iterations=100)
        # pos = nx.spring_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(g)

    # nx.draw_networkx(G, with_labels=True, node_size=2, width=0.15, font_size = 1.5, node_color=colors,pos=pos)
    # nx.draw_networkx(G, with_labels=False, node_size=1.5, width=0.2, font_size = 1.5, linewidths=0.2, node_color = 'k',pos=pos,alpha=0.2)

    nx.draw_networkx_nodes(g, pos, cmap=cmap, node_size=node_size, node_color='#336699', alpha=1, linewidths=0)
    nx.draw_networkx_edges(g, pos, alpha=alpha, width=width)

    plt.axis('off')
    # plt.title('Complete Graph of Odd-degree Nodes')
    # plt.show()

    plt.tight_layout()

    print("Saving ", file_name)
    plt.imsave
    plt.savefig(file_name, dpi=600)
    plt.close()


def plot_metrics():
    """

    :return:
    """
    sns.set()
    sns.set_style("ticks")
    sns.set_context("poster", font_scale=1.28, rc={"lines.linewidth": 3})

    # plot robustness result
    noise = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    MLP_degree = np.array([0.3440, 0.1365, 0.0663, 0.0430, 0.0214, 0.0201])
    RNN_degree = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    BA_degree = np.array([0.0892, 0.3558, 1.1754, 1.5914, 1.7037, 1.7502])
    Gnp_degree = np.array([1.7115, 1.5536, 0.5529, 0.1433, 0.0725, 0.0503])

    MLP_clustering = np.array([0.0096, 0.0056, 0.0027, 0.0020, 0.0012, 0.0028])
    RNN_clustering = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    BA_clustering = np.array([0.0255, 0.0881, 0.3433, 0.4237, 0.6041, 0.7851])
    Gnp_clustering = np.array([0.7683, 0.1849, 0.1081, 0.0146, 0.0210, 0.0329])

    plt.plot(noise, Gnp_degree)
    plt.plot(noise, BA_degree)
    plt.plot(noise, MLP_degree)
    # plt.plot(noise, RNN_degree)

    # plt.rc('text', usetex=True)
    plt.legend(['E-R', 'B-A', 'GraphRNN'])
    plt.xlabel('Noise level')
    plt.ylabel('MMD degree')

    plt.tight_layout()
    plt.savefig('figures_paper/robustness_degree.png', dpi=300)
    plt.close()

    # plt.plot(noise, Gnp_clustering)
    plt.plot(noise, BA_clustering)
    plt.plot(noise, MLP_clustering)
    # plt.plot(noise, RNN_clustering)
    plt.legend(['E-R', 'B-A', 'GraphRNN'])
    plt.xlabel('Noise level')
    plt.ylabel('MMD clustering')

    plt.tight_layout()
    plt.savefig('figures_paper/robustness_clustering.png', dpi=300)
    plt.close()


def imsave(image_file_name, image, v_min=None, v_max=None, c_map=None, img_format=None, img_dpi=1, origin=None):
    """

    :param image_file_name:   image name
    :param image:
    :param v_min:
    :param v_max:
    :param c_map:
    :param img_format:
    :param origin:
    :return:
    """
    fig = Figure(figsize=image.shape[::-1], dpi=1, frameon=False)
    fig.figimage(image, cmap=c_map, vmin=v_min, vmax=v_max, origin=origin)
    fig.savefig(image_file_name, img_dpi=1, format=img_format)


def save_prediction_histogram(prediction, file_name, max_depth, bin_size=20):
    """

    :param prediction:
    :param file_name:
    :param max_depth:
    :param bin_size:
    :return:
    """
    bin_edge = np.linspace(1e-6, 1, bin_size + 1)
    output_pred = np.zeros((bin_size, max_depth))
    for i in range(max_depth):
        output_pred[:, i], _ = np.histogram(prediction[:, i, :], bins=bin_edge, density=False)
        output_pred[:, i] /= np.sum(output_pred[:, i])
    imsave(fname=file_name, arr=output_pred, origin='upper', cmap='Greys_r', vmin=0.0, vmax=3.0 / bin_size)
