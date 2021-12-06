import concurrent.futures
import os
import subprocess as sp
from datetime import datetime

import networkx
import networkx as nx
import numpy as np

from .mmd import compute_mmd
from .mmd import gaussian
from .mmd import gaussian_emd
import traceback
from typing import List
from typing import List
from typing import List, Set, Dict, Tuple, Optional
from typing import Callable, Iterator, Union, Optional, List
from typing import Union, Any, List, Optional, cast
from typing import AnyStr


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x + y


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, trace=False):
    """ Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    print(len(sample_ref), len(sample_pred))
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elapsed = datetime.now() - prev
    if trace:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def clustering_worker(param):
    """

    """
    G, bins = param
    hist, _ = np.histogram(list(nx.clustering(G).values()), bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True, is_trace=True):
    """

    """
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    elapsed = datetime.now() - prev

    if is_trace:
        print('Time computing clustering mmd: ', elapsed)

    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph, default_location_orca='./external/graphrnn_eval/orca/',
         orca_output='external/graphrnn_eval/orca/tmp.txt'):
    """
    mbayaramo fixed bunch of staff

    - compiled with visual studio 2019
    - pybind fixed
    - subproccess fixed

    """
    tmp_fname = orca_output
    # file = codecs.open("temp", "w", "utf-8")

    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    if os.name == 'nt':
        output = sp.check_output(['./external/graphrnn_eval/orca/orca.exe', 'node', '4',
                                  'external/graphrnn_eval/orca/tmp.txt', 'std'],
                                   encoding="utf-8", universal_newlines=True)
    else:
        output = sp.check_output(['./external/graphrnn_eval/orca/orca', 'node', '4',
                                  'external/graphrnn_eval/orca/tmp.txt', 'std'])
        output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(graph_ref_list: List[networkx.classes.graph.Graph],
                graph_pred_list: List[networkx.classes.graph.Graph],
                motif_type='4cycle', ground_truth_match=None, bins=100):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)
        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        """
        """
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False)

    # print('-------------------------')
    # print(np.sum(total_counts_ref) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred) / len(total_counts_pred))
    # print('-------------------------')
    return mmd_dist


def orbit_stats_all(graph_ref_list: List[networkx.classes.graph.Graph],
                    graph_pred_list: List[networkx.classes.graph.Graph], debug_orca=True):
    """
    Details here
    (Refactored bunch of staff to make all this staff to work)
    #https://github.com/qema/orca-py
    """
    if len(graph_ref_list) == 0 or len(graph_pred_list) == 0:
        return

    total_counts_ref = []
    total_counts_pred = []

    # TODO not sure why they do that
    # graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except Exception as e:
            if debug_orca:
                print(e)
                traceback.print_exc()
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except Exception as e:
            if debug_orca:
                print(e)
                traceback.print_exc()
            continue

        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    mmd_dist = compute_mmd(total_counts_ref,
                           total_counts_pred,
                           kernel=gaussian,
                           is_hist=False, sigma=30.0)

    #
    # print('-------------------------')
    # print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    # print('-------------------------')

    return mmd_dist
