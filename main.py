# Model based on GraphRNN
# Author Mustafa Bayramov
import argparse
import random
import sys
import time
from datetime import time
from datetime import timedelta
from typing import Final

import numpy as np
import torch
import torch.utils as tutil

from shapegnet import create_graphs
from shapegnet.external.graphrnn_eval.stats import degree_stats, clustering_stats, orbit_stats_all
from shapegnet.generator_trainer import GeneratorTrainer
from shapegnet.model_config import ModelSpecs
from shapegnet.model_creator import ModelCreator
from shapegnet.models.adjacency_decoder import AdjacencyDecoder
from shapegnet.models.sampler.GraphSeqSampler import GraphSeqSampler
from shapegnet.plotlib import plot
from shapegnet.plotlib.plot import draw_single_graph
from shapegnet.utils import fmt_print, fmtl_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN: Final = 1
TEST: Final = 2
PREDICTION: Final = 3


def draw_samples(trainer_spec: ModelSpecs,
                 num_samples=10):
    """
    If model already trained and generated prediction
    plot prediction.
    """
    fmtl_print("Train graph", trainer_spec.get_active_train_graph())
    fmtl_print("Number of prediction files", len(trainer_spec.get_active_model_prediction_files()))
    fmtl_print("Last saved epoch", trainer_spec.get_last_saved_epoc())

    last_saved_epoch = trainer_spec.get_last_saved_epoc()['node_model']
    graphs = trainer_spec.get_last_graph_stat()

    dict_dup = {}
    for i in range(0, num_samples):
        # random id for a graph,
        # by default we sample from last epoch only and no duplicates
        gid = random.randint(0, len(graphs))
        if gid in dict_dup:
            continue
        dict_dup[gid] = gid
        file_name = trainer_spec.generate_prediction_figure_name(last_saved_epoch, sample_time=1, gid=gid)
        draw_single_graph(graphs[gid], file_name=file_name, plot_type='prediction', graph_name="test")


def graph_completion(g, p=0.5):
    """
    """
    for v in g:
        # remove node
        for node in list(v.nodes()):
            if np.random.rand() > p:
                v.remove_node(node)
        # remove edge
        for e in list(v.edges()):
            # print('edge',edge)
            if np.random.rand() > p:
                v.remove_edge(e[0], e[1])


def compute_graph_split_len(gv, gt):
    """
    Compute split based on number of nodes and edges

    and return normalized value.
    @param gv:
    @param gt:
    @return:
    """

    return sum(g.number_of_nodes() for g in gv) / len(gv), \
           sum(g.number_of_nodes() for g in gt) / len(gt)


def generate_train_test(g, specs: ModelSpecs, is_fix_seed=True, is_shuffled=True):
    """
    Generate test , train , validation split
    """
    # split datasets
    if is_fix_seed:
        random.seed(123)

    if is_shuffled:
        random.shuffle(g)

    graphs_len = len(g)
    return g[int(specs.test_ratio() * graphs_len):], \
           g[0:int(specs.train_ratio() * graphs_len)], \
           g[0:int(specs.validation_ratio() * graphs_len)]


def prepare(trainer_spec):
    """
    Prepare dir , clean up etc.
    """
    trainer_spec.build_dir()
    trainer_spec.setup_tensorflow()


def create_dataset_sampler(trainer_spec: ModelSpecs, graphs, num_workers=None):
    """

    @param trainer_spec: trainer specification, include strategy how to sample ration etc.
    @param graphs: a graph that we use to train network
    @param num_workers:
    @return: return torch.util.data.DataLoader
    """
    # dataset initialization
    if trainer_spec.max_prev_node() > 0:
        dataset = GraphSeqSampler(graphs,
                                  max_prev_node=trainer_spec.max_prev_node(),
                                  max_num_node=trainer_spec.max_num_node())
    else:
        dataset = GraphSeqSampler(graphs)

    normalized_weight = [1.0 / len(dataset) for i in range(len(dataset))]
    sample_strategy = tutil.data.sampler.WeightedRandomSampler(normalized_weight,
                                                               num_samples=trainer_spec.compute_num_samples(),
                                                               replacement=True)

    _num_workers = trainer_spec.num_workers()
    if num_workers is not None:
        _num_workers = num_workers

    dataset_loader = tutil.data.DataLoader(dataset,
                                           batch_size=trainer_spec.batch_size(),
                                           num_workers=_num_workers,
                                           sampler=sample_strategy,
                                           pin_memory=False)

    return dataset_loader


def clean_graphs(graph_real, graph_pred, is_shuffle=True):
    """
    Selecting graphs generated that have the similar sizes.
    It is usually necessary for GraphRNN-S version, but not the full GraphRNN model.
    """

    #
    if is_shuffle:
        random.shuffle(graph_real)
        random.shuffle(graph_pred)

    # get length
    real_graph_len = np.array([len(graph_real[i]) for i in range(len(graph_real))])
    pred_graph_len = np.array([len(graph_pred[i]) for i in range(len(graph_pred))])

    fmt_print("Real graph size", real_graph_len)
    fmt_print("Prediction graph size", pred_graph_len)

    # # select pred samples
    # # The number of nodes are sampled from the similar distribution as the training set
    # pred_graph_new = []
    # pred_graph_len_new = []
    # for value in real_graph_len:
    #     pred_idx = find_nearest_idx(pred_graph_len, value)
    #     pred_graph_new.append(graph_pred[pred_idx])
    #     pred_graph_len_new.append(pred_graph_len[pred_idx])
    # return graph_real, pred_graph_new


def compute_generic_stats(graph):
    """
    Compute generic statistic for graph
    :param graph:
    :return:
    """
    num_nodes = 0
    for g in graph:
        num_nodes += g.number_of_nodes()

    if len(graph) > 0:
        num_nodes /= len(graph)
        fmtl_print('Average number of nodes (graph size {})'.format(len(graph)), num_nodes)


def evaluate(cmds, trainer_spec: ModelSpecs,
             epoch_start=1,
             epoch_step=1):

    # get a graphs
    try:
        train_graph, saved_in_test = trainer_spec.load_train_test()
        graph_test_len = len(train_graph)
    except FileNotFoundError:
        print("No graph file found.")
        return

    # x_df = pd.DataFrame(x_np)
    # x_np = x.numpy()
    last_saved_epoch = trainer_spec.get_last_saved_epoc()['node_model']
    validate_set = saved_in_test[0:int(0.2 * graph_test_len)]
    test_set = saved_in_test[int(0.8 * graph_test_len):]

    compute_generic_stats(test_set)
    fmt_print("Last saved epoch", last_saved_epoch)
    predictions = trainer_spec.get_prediction_graph()
    for i, (file_name, epoch_predicted) in enumerate(predictions):
        if i < last_saved_epoch:
            continue
        fmtl_print("Processing {}".format(i), file_name)

        #
        compute_generic_stats(epoch_predicted)
        compute_generic_stats(validate_set)
        clean_graphs(test_set, epoch_predicted)

        # evaluate mmd test, between prediction and test
        mmd_degree = -1
        if trainer_spec.mmd_degree():
            mmd_degree = degree_stats(test_set, epoch_predicted)
            mmd_degree_validate = degree_stats(test_set, validate_set)
            fmt_print('Train/Generated MMD', mmd_degree)
            fmt_print('Train/Validation MMD', mmd_degree_validate)

        # #
        # mmd_clustering = -1
        # if trainer_spec.mmd_clustering():
        #     mmd_clustering = clustering_stats(graph_in_test, epoch_predicted)
        #     fmt_print('Graph clustering', mmd_clustering)
        # #
        # mmd_4orbits = -1
        # if trainer_spec.mmd_orbits():
        #     mmd_orbits = orbit_stats_all(graph_in_test, epoch_predicted)
        #     fmt_print('Graph orbit', mmd_4orbits)

        # x_np = [i] = [mmd_degree, mmd_clustering, mmd_orbits]
        #print('degree', mmd_degree, 'clustering', mmd_clustering, 'orbits', mmd_4orbits)

    # x_df.to_csv('tmp.csv')


def main_train(cmds, trainer_spec: ModelSpecs):
    """

    """
    # prepare test environment
    prepare(trainer_spec)

    # create model creator
    model_creator = ModelCreator(trainer_spec, device)

    # model graph specs
    print("###############################################")
    fmtl_print("Creating graphs type", trainer_spec.active)
    fmtl_print("Maximum previous node to track", trainer_spec.max_prev_node())
    fmtl_print("Maximum nodes to track", trainer_spec.max_num_node())

    # create dataset based on specs in config.yaml
    graphs = create_graphs.create(trainer_spec)
    max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])

    trainer_spec.trainer_spec = trainer_spec.set_max_num_node(max_num_node)
    #
    fmtl_print('traing/test/val ratio', trainer_spec.train_ratio(),
               trainer_spec.test_ratio(), trainer_spec.validation_ratio())
    fmtl_print('max previous node', trainer_spec.max_prev_node())
    fmtl_print('max number node', trainer_spec.max_num_node())
    fmtl_print('max/min number edge', max_num_edge, min_num_edge)

    # compute splits
    graphs_test, graphs_train, graphs_validate = generate_train_test(graphs, trainer_spec)
    graph_validate_len, graph_test_len = compute_graph_split_len(graphs_validate, graphs_test)
    fmtl_print('total/train/test/validate sizes', len(graphs), len(graphs_train), len(graphs_test),
               len(graphs_validate))
    fmtl_print('validation/test', graph_validate_len, graph_test_len)
    fmtl_print('total graph number, training subset', len(graphs), len(graphs_train))
    print("###############################################")

    # load_pretrained(trainer_spec)
    # save ground truth graphs
    # To get train and test set, after loading you need to manually slice
    GeneratorTrainer.save_graphs(graphs, str(trainer_spec.train_graph_file()))
    GeneratorTrainer.save_graphs(graphs, str(trainer_spec.test_graph_file()))

    # plot training set if needed
    if cmds.plot == 'train':
        plot.draw_samples_from_file(trainer_spec.train_graph_file(), plot_type='train',
                                    file_prefix=trainer_spec.train_plot_filename(),
                                    num_samples=10)

    dataset_loader = create_dataset_sampler(trainer_spec, graphs_train)

    if training_spec.is_read_benchmark():
        read_start_timer = time.monotonic()
        for _, _ in enumerate(dataset_loader):
            pass
        read_stop_timer = time.monotonic()
        fmt_print("Dataset read time", timedelta(seconds=read_stop_timer - read_start_timer), "sec")
        return

    models = model_creator.create_model(verbose=True)
    if trainer_spec.is_train_network():
        decoder = AdjacencyDecoder()
        trainer = model_creator.create_trainer(dataset_loader, models, decoder)
        trainer.set_notebook(False)
        trainer.set_verbose(False)
        trainer.train()


def main(cmds, trainer_spec: ModelSpecs):
    """
    Main entry
    @param cmds:
    @param trainer_spec:
    @return:
    """

    # draw at the end
    if trainer_spec.is_train_network():
        main_train(cmds, trainer_spec)

    # draw at the end
    if trainer_spec.is_draw_samples():
        if not trainer_spec.is_trained():
            sys.exit("Check configuration file,  it looks like model {} "
                     "is untrained.".format(trainer_spec.get_active_model()))
        draw_samples(trainer_spec)

    # evaluate statistic
    if trainer_spec.is_evaluate():
        if not trainer_spec.is_trained():
            sys.exit("Check configuration file,  it looks like model {} "
                     "is untrained.".format(trainer_spec.get_active_model))
        evaluate(cmds, trainer_spec)


if __name__ == '__main__':
    """
    """
    fmtl_print("Torch cudnn backend version: ", torch.backends.cudnn.version())
    parser = argparse.ArgumentParser(description="Trains a graph generator models.")
    parser.add_argument("--device", default="cuda", help="[cpu,cuda]")
    parser.add_argument("--plot", default="none", help="plot train or test set")
    parser.add_argument("--hidden_size", default=100, type=int, help="number of hidden units in each flow layer", )
    parser.add_argument("--n_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--n_samples", default=30000, type=int, help="total number of data points in toy dataset", )

    # parse args and read config.yaml
    cmd = parser.parse_args()
    training_spec = ModelSpecs()

    fmtl_print("Model in training mode", training_spec.is_train_network())
    fmtl_print("Model in evaluate mode", training_spec.is_evaluate())
    fmtl_print("Model in generate sample", training_spec.is_draw_samples())
    fmtl_print("Model active dataset", training_spec.active)
    fmtl_print("Model active dataset", training_spec.epochs())
    fmtl_print("Model active dataset", training_spec.batch_size())
    fmtl_print("Model number of layers", training_spec.num_layers())
    fmtl_print("Active model", training_spec.active_model)

    # run
    main(cmd, training_spec)