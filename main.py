# ShapeGen
#
# This project explores the problem of synthetic graph and mesh generation in
# auto-regressive and adversarial nets settings.
#
# In both cases, my aim to generate realistic graphs and analyze how a model can generalize.
#
#
# Author Mustafa Bayramov
import argparse
import random
import sys
import time
from datetime import time
from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.utils as tutil
import seaborn as sns

from shapegnet import create_graphs
from shapegnet.external.graphrnn_eval.stats import degree_stats, clustering_stats, orbit_stats_all
from shapegnet.generator_trainer import GeneratorTrainer
from shapegnet.model_config import ModelSpecs
from shapegnet.model_creator import ModelCreator
from shapegnet.models.adjacency_decoder import AdjacencyDecoder
from shapegnet.models.sampler.GraphSeqSampler import GraphSeqSampler
from shapegnet.plotlib import plot
from shapegnet.plotlib.plot import draw_single_graph
from shapegnet.utils import fmt_print, fmtl_print, find_nearest
import pandas as pd
import matplotlib as plt
from datetime import datetime

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
    Create dataset , dataset sampler based on trainer specification.

    @param trainer_spec: trainer specification, include strategy how to sample ration etc.
    @param graphs: a graph that we use to train network.
    @param num_workers:
    @return: return torch.util.data.DataLoader
    """
    # dataset initialization
    if trainer_spec.max_depth() > 0:
        dataset = GraphSeqSampler(graphs,
                                  max_depth=trainer_spec.max_depth(),
                                  max_nodes=trainer_spec.max_nodes())
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


def select_graph(real_graph, generated_graph, is_shuffle=True):
    """
        We selecting graphs generated that have the similar sizes.

    @param real_graph:
    @param generated_graph:
    @param is_shuffle:
    @return:
    """

    #
    if is_shuffle:
        random.shuffle(real_graph)
        random.shuffle(generated_graph)

    # get length
    real_graph_len = np.array([len(real_graph[i]) for i in range(len(real_graph))])
    pred_graph_len = np.array([len(generated_graph[i]) for i in range(len(generated_graph))])

    fmt_print("Real graph size", real_graph_len)
    fmt_print("Prediction graph size", pred_graph_len)

    # select pred samples
    pred_graph_new = []
    pred_graph_len_new = []
    for value in real_graph_len:
        pred_idx = find_nearest(pred_graph_len, value)
        pred_graph_new.append(generated_graph[pred_idx])
        pred_graph_len_new.append(pred_graph_len[pred_idx])

    return real_graph, pred_graph_new


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


def evaluate_prediction(trainer_spec: ModelSpecs,
                        is_verbose=False,
                        last_epoch_only=False,
                        default_loc=None):
    sns.set()

    # get a graphs
    try:
        train_graph, saved_in_test = trainer_spec.load_train_test()
        graph_test_len = len(train_graph)
    except FileNotFoundError:
        print("No graph file found.")
        return

    last_saved_epoch = trainer_spec.get_last_saved_epoc()['node_model']
    validate_set = saved_in_test[0:int(0.2 * graph_test_len)]
    test_set = saved_in_test[int(0.8 * graph_test_len):]

    compute_generic_stats(test_set)
    if last_epoch_only:
        predictions = trainer_spec.get_prediction_graph(is_last=last_epoch_only)
    else:
        predictions = trainer_spec.get_prediction_graph()

    mmd_degree_predict = []
    mmd_degrees_validates = []
    mmd_clustering_predict = []
    mmd_clustering_validate = []
    mmd_orbits_predict = []
    mmd_orbits_validate = []
    epochs = []

    compute_generic_stats(test_set)
    fmt_print("Last saved epoch", last_saved_epoch)

    for i, (epoch, sample_time, file_name, epoch_predicted) in enumerate(predictions):

        if last_epoch_only is True and i < last_saved_epoch:
            continue

        if is_verbose is True:
            fmtl_print("Processing epoch {} sample {} file_name".format(epoch, sample_time), file_name)

        epochs.append(epoch)
        #
        compute_generic_stats(epoch_predicted)
        compute_generic_stats(validate_set)
        select_graph(test_set, epoch_predicted)

        indices = np.random.randint(low=0, high=len(epoch_predicted), size=(20,))
        predication_selection = []
        for (p, j) in enumerate(indices):
            predication_selection.append(epoch_predicted[j])

        # evaluate mmd test, between prediction and test
        mmd_degree = -1
        if trainer_spec.mmd_degree():
            mmd_degree = degree_stats(test_set, predication_selection)
            mmd_degree_validate = degree_stats(test_set, validate_set)
            fmtl_print('Degree train/Generated MMD', mmd_degree)
            fmtl_print('Degree train/Validation MMD', mmd_degree_validate)
            mmd_degree_predict.append(mmd_degree.copy())
            mmd_degrees_validates.append(mmd_degree_validate.copy())
        #
        mmd_clustering = -1
        if trainer_spec.mmd_clustering():
            mmd_clustering_a = clustering_stats(test_set, predication_selection)
            mmd_clustering_b = clustering_stats(test_set, validate_set)
            fmtl_print('Clustering train/Generated MMD', mmd_clustering_a)
            fmtl_print('Clustering train/Validation MMD', mmd_clustering_b)
            mmd_clustering_predict.append(mmd_clustering_a.copy())
            mmd_clustering_validate.append(mmd_clustering_b.copy())

        list_of_tuples = list(
            zip(epochs, mmd_degree_predict, mmd_degrees_validates, mmd_clustering_predict,
                mmd_clustering_validate))

        # print(list_of_tuples)
        df = pd.DataFrame(list_of_tuples, columns=['epoch',
                                                   'pdegree',
                                                   'vdegree',
                                                   'pcluster',
                                                   'vcluster'])

        ts = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        if default_loc is None:
            file_name = 'eval_stats' + ts + '.csv'
            df.to_csv(file_name)
        else:
            _path = Path(default_loc) / 'eval_stats'
            df.to_csv(_path + ts + '.csv')

        return df


def evaluate(cmds, trainer_spec: ModelSpecs,
             last_epoch_only=False,
             default_loc=None):
    frame = evaluate_prediction(trainer_spec,
                                last_epoch_only=last_epoch_only,
                                default_loc=default_loc)
    df = frame.astype(float)
    # df.plot(kind='line', x='vdegree', y='pdegree', ax=ax)

    plt.style.use('ggplot')

    x1 = pd.Series(df['pdegree'], name='pred_degree')
    x2 = pd.Series(df['vdegree'], name='val_degree')

    sns.pairplot(df,
                 x_vars="epoch",
                 y_vars=["vdegree", "pdegree"],)


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
    fmtl_print("Maximum previous node to track", trainer_spec.max_depth())
    fmtl_print("Maximum nodes to track", trainer_spec.max_nodes())

    # create dataset based on specs in config.yaml
    graphs = create_graphs.create(trainer_spec)
    max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])

    trainer_spec.trainer_spec = trainer_spec.set_max_num_node(max_num_node)
    #
    fmtl_print('traing/test/val ratio', trainer_spec.train_ratio(),
               trainer_spec.test_ratio(), trainer_spec.validation_ratio())
    fmtl_print('max previous node', trainer_spec.max_depth())
    fmtl_print('max number node', trainer_spec.max_nodes())
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
    if cmds is not None and cmds.plot_training:
        plot.draw_samples_from_file(trainer_spec.train_graph_file(),
                                    plot_type='train',
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    """
    """
    fmtl_print("Torch cudnn backend version: ", torch.backends.cudnn.version())
    parser = argparse.ArgumentParser(description="Trains a graph generator models.")
    parser.add_argument("--device", default="cuda", help="[cpu,cuda]")
    parser.add_argument("--no_plot", default=False, type=bool, help="Disables plot after model trained.")
    parser.add_argument("--verbose", default=False, type=bool, help="Enables verbose output.")
    parser.add_argument("--train", default=False, type=bool, help="Train a network.")
    parser.add_argument("--plot_training", default=False, type=bool, help="Enable plotting example from train set.")

    parser.add_argument("--n_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--n_samples", default=30000, type=int, help="total number of data points in toy dataset", )

    # parse args and read config.yaml
    cmd = parser.parse_args()
    training_spec = ModelSpecs()
    if 'no_plot' in cmd and cmd.no_plot is True:
        training_spec.set_draw_samples(False)

    if cmd.verbose:
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
