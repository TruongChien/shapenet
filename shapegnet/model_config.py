# Graph Generator Model Configurator
#
# All trainer parameters abstracted in separate entity.
#
#  - Trainer , Evaluator and the rest this class to configurator and store data.
#     It read yaml config file that users passes either as file name or io.string.
#
# Mustafa B
import os
import sys
from os import listdir
from os.path import isfile, join
import shutil
import logging
from pathlib import Path
from time import strftime, gmtime
import networkx as nx
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
# this is main colab fix.
from typing import List
from typing import List
from typing import List, Set, Dict, Tuple, Optional
from typing import Callable, Iterator, Union, Optional, List
from typing import Union, Any, List, Optional, cast
from typing import AnyStr

from .graph_tools import graph_from_file
from .utils import fmt_print, fmtl_print

_opt_example = """"
optimizer:
  - name: node_optimizer
    eps: 1e-8
    weight_decay: 0
    amsgrad: False
    momentum=0:
    betas: [0.9, 0.999]
    type: Adam
"""


def extract_value_from_file(f, key, file_type='.dat'):
    """
    @param file_type:
    @param f:
    @param key:
    @return:
    """
    proceed = f.split('_')
    for i, k in enumerate(proceed):
        if key in k:
            v = proceed[i + 1]
            if file_type in v:
                return v[:-len(file_type)]
            else:
                return v


class ModelSpecs:
    """
    The class hold all trainer configuration settings.

    """

    def __init__(self, template_file_name='config.yaml', verbose=False):
        """

        :param template_file_name:
        :param verbose:
        """
        if isinstance(template_file_name, str):
            fmtl_print("Loading", template_file_name)

        # store point to current loaded config,
        # after spec read serialize yaml to it.
        self.config = None

        # device
        self.device = 'cuda'
        fmtl_print("Device", self.device)

        # a file name or io.string
        self.config_file_name = template_file_name

        # if clean tensorboard
        self.clean_tensorboard = False

        # Cuda device id , if device is cuda
        self.cuda = 1

        # indicate active template to use
        self.active = ""

        # model spec used
        self.graph_specs = None

        # dictionary of models
        self.models = None

        # active model
        self.active_model = None

        # pointer to active model
        self.model = None

        # list of of lr scheduler
        self.lr_schedulers = None

        # list of optimizer defined in config file
        self._optimizers = None

        # tensorboard writer
        self.writer = None

        # Which graph dataset is used to train the model
        self.graph_type = self.active

        # verbose mode will output data to console
        self._verbose = verbose

        # stores a setting type ( name ), it must be defined in config.yaml
        self._active_setting = None

        # stores current global settings,
        # list of setting defined in config.yaml
        self._setting = None

        # will be pre-computed
        self._batch_ratio = None

        # initialized
        self.inited = False

        # if argument is filename, read config.
        if isinstance(template_file_name, str):
            if len(template_file_name) == 0:
                raise Exception("path to config.yaml file is empty.")
            self.read_from_file()
        else:
            self.read_from_stream(template_file_name)

        # hidden size for main RNN
        self.hidden_size_rnn = int(128 / self.parameter_shrink())

        # hidden size for output RNN
        self.hidden_size_rnn_output = 16

        # the size for LSTM input
        self.embedding_size_rnn = int(64 / self.parameter_shrink())

        # the embedding size for edge rnn
        self.embedding_size_rnn_output = 8

        # the embedding size for output
        self.embedding_size_output = int(64 / self.parameter_shrink())

        # training config
        self.rescale_factor = 10

        # output dirs
        self.dir_input = self.root_dir()
        self.dir_result = Path(self.dir_input) / Path(self.results_dir())
        self.dir_log = Path(self.dir_result) / Path(self.log_dir())
        self.model_save_path = self.dir_result / Path(self.model_save_dir())
        self.dir_graph_save = self.dir_result / Path(self.graph_dir())
        self.dir_figure = self.dir_result / Path(self.figures_dir())
        self.dir_timing = self.dir_result / Path(self.timing_dir())
        # default dir where we store serialized prediction graph as image
        self.dir_model_prediction = self.dir_result / Path(self.prediction_dir())
        # self.dir_model_prediction = self.dir_result / Path(self.figures_prediction_dir())

        # filenames to save results, statistics , traces , logs
        self.filename = None
        self.filename_prediction = None
        self.filename_train = None
        self.filename_test = None
        self.filename_metrics = None
        self.filename_time_traces = None

        # self.filename_baseline = self.dir_graph_save / Path(
        #     self.graph_type + self.generator_baseline + '_' + self.metric_baseline)

        # generate all template file names
        self.generate_file_name_template()

        # fmt for filenames
        self._epoch_file_fmt = "epoch"

    def get_prediction_dir(self):
        """
        Return directory where prediction stored.
        :return:
        """
        return self.dir_model_prediction

    def generate_file_name_template(self):
        """
         Generates file name templates.
        """
        self.filename = "{}_{}_{}_layers_{}_hidden_{}_".format(self.active,
                                                               self.active_model,
                                                               self.graph_type,
                                                               str(self.num_layers()),
                                                               str(self.hidden_size_rnn))

        self.filename_prediction = self.filename + 'predictions_'
        self.filename_train = self.filename + 'train_'
        self.filename_test = self.filename + 'test_'
        self.filename_metrics = self.filename + 'metric_'
        self.filename_time_traces = self.filename + 'timetrace_'

    def template_file_name(self):
        """
        Return main template file name,  template file
        name contains model name and other details.
        :return:
        """
        return self.filename

    def prediction_dir_from_type(self, file_type):
        """
        Return prediction dir based on file type
        :param file_type:  dat or png
        :return: return prediction dir based on file type
        """
        if file_type == 'dat':
            return self.dir_model_prediction
        elif file_type == 'png':
            return self.dir_figure
        else:
            raise Exception("Unknown format for prediction.")

    def prediction_filename(self, epoch=None, sample=None, gid=None, file_type='dat'):
        """
        Return prediction file name
        @param epoch:
        @param sample:
        @param file_type:
        @param gid:
        @return:
        """
        _dir = self.prediction_dir_from_type(file_type)

        if epoch is not None:
            if gid is None:
                return str(_dir / Path(self.filename_prediction
                                       + 'epoch' + '_' + str(epoch) + '_'
                                       + 'sample' + '_' + str(sample) + '.' + file_type))
            else:
                return str(_dir / Path(self.filename_prediction
                                       + 'epoch' + '_' + str(epoch) + '_'
                                       + 'sample' + '_' + str(sample) +
                                       '_gid_' + str(gid) + '.' + file_type))

        return self.filename_prediction

    def train_filename(self, epoch=None, sample=None, file_type='dat'):
        """
        Return file name used for training
        :param epoch:
        :param sample:
        :param file_type:
        :return:
        """
        if epoch is not None:
            return str(self.dir_graph_save / Path(self.filename_train
                                                  + 'epoch' + '_' + str(epoch) + '_'
                                                  + 'sample' + '_' + str(sample) + '.' + file_type))
        return self.filename_train

    def train_plot_filename(self, epoch=None, sample=None):
        """

        """
        return str(self.dir_figure / Path(self.filename_train))

    def test_filename(self, epoch=None, sample=None, file_type='dat'):
        """

        """
        if epoch is not None:
            return str(self.dir_graph_save / Path(self.filename_test
                                                  + 'epoch' + '_' + str(epoch) + '_'
                                                  + 'sample' + '_' + str(sample) + '.' + file_type))
        return self.filename_test

    def train_graph_file(self):
        """
        Default generated graphs state file.
        """
        return self.train_filename(epoch=0, sample=self.sample_time())

    def test_graph_file(self):
        """
        Default generated test state file.
        """
        return self.test_filename(epoch=0, sample=self.sample_time())

    def save_timed_trace(self, epoch=None, sample=None, file_type='dat') -> str:
        """

        """
        if epoch is not None:
            return str(self.dir_timing / Path(self.filename_time_traces
                                              + 'epoch' + '_' + str(epoch) + '_'
                                              + 'sample' + '_' + str(sample) + '.' + file_type))
        return self.filename_time_traces

    def model_node_file_name(self) -> str:
        """
        Models checkpoint filename.
        @return:
        """
        files_dict = self.model_filenames()
        if 'node_model' in files_dict:
            return files_dict['node_model']

        return str(self.dir_graph_save / Path('default_train_node_rnn_state.dat'))

    def model_edge_file_name(self) -> str:
        """
        Models checkpoint filename.
        @return:
        """
        files_dict = self.model_filenames()
        if 'edge_model' in files_dict:
            return files_dict['edge_model']

        return str(self.dir_graph_save / Path('default_train_edge_rnn_state.dat'))

    def models_list(self):
        """
        List of network types and sub-network models used for a given model.
        For example GraphRNN has graph edge and graph node model.
        @return: list of models.
        """
        models_types = []
        for k in self.model:
            if k.find('model') != -1:
                models_types.append(k)

    def model_filenames(self, file_type='.dat'):
        """

        Returns dict that hold sub-model name and
        respected checkpoint filename.

        @param file_type:
        @return:
        """
        models_filenames = {}
        for k in self.model:
            if k.find('model') != -1:
                models_filenames[k] = str(self.model_save_path /
                                          Path(self.filename + '_' + k + '_' + str(self.load_epoch()) + file_type))
        return models_filenames

    def is_dropout(self) -> bool:
        """
        TODO
        """
        return bool(self.get_config_key('is_dropout'))

    def get_optimizer(self, alias_name: str):
        """
        Method return optimizer setting.
        :param alias_name: alias name in config.  It bind optimizer to model
        :return: dict that hold optimizer settings
        """
        return self._optimizers[alias_name]

    def read_config(self, debug=False):
        """

        Parse config file and initialize trainer

        :param debug: will output debug into
        :return: nothing
        """
        if debug:
            fmtl_print("Parsing... ", self.config)

        if 'active' not in self.config:
            raise Exception("config.yaml must contains valid active settings.")
        self.active = self.config['active']

        if 'graph' not in self.config:
            raise Exception("config.yaml must contains corresponding graph settings for {}".format(self.active))

        g_list = self.config['graph']
        if self.active not in g_list:
            raise Exception("config.yaml doesn't contain {} template, check config.".format(self.active))

        self.graph_specs = self.config['graph'][self.active]

        if 'models' not in self.config:
            raise Exception("config.yaml must contain at least one models list and one model.")

        if 'use_model' not in self.config:
            raise Exception("config.yaml must contain use_model and it must defined.")

        self.active_model = self.config['use_model']
        self.models = self.config['models']

        if self.active_model not in self.models:
            raise Exception("config.yaml doesn't contain model {}.".format(self.active_model))

        self.model = self.models[self.active_model]

        if 'lr_schedulers' in self.config:
            self.lr_schedulers = self.config['lr_schedulers']

        # checks if optimizers setting present
        if 'optimizers' in self.config:
            self._optimizers = self.config['optimizers']
        else:
            raise Exception("config.yaml doesn't contain optimizers section. Example {}".format(_opt_example))

        if 'settings' not in self.config:
            raise Exception("config.yaml must contain at least one section with a global setting.")

        if 'active_setting' not in self.config:
            raise Exception("config.yaml must contain name of the "
                            "global setting that the trainer must use.")

        # settings stored internally
        if debug:
            fmt_print("active setting", self.config['active_setting'])
        self._active_setting = self.config['active_setting']

        _settings = self.config['settings']
        if debug:
            fmt_print("Settings list", _settings)

        self._setting = _settings[self._active_setting].copy()
        if debug:
            fmt_print("Active settings", self._setting)

        self.inited = True

    def read_from_file(self, debug=False):
        """
        Read config file and initialize trainer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(self.config_file_name, "r") as stream:
            try:
                fmtl_print("Reading... ", self.config_file_name)
                self.config = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

        self.read_config()

    def read_from_stream(self, buffer, debug=False):
        """
        Read config file from a stream
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            if self._verbose:
                print("Reading from io buffer")
            self.config = yaml.load(buffer, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit("Failed parse yaml")

        self.read_config()

    def get_config_key(self, key):
        """
        Return generic config element by key
        """
        if key in self.config:
            return self.config[key]

    def epochs(self) -> int:
        """
        Return epochs,
        Note each graph has own total epochs. ( depend on graph size)
        :return: number of epochs to run for given dataset, default 100
        """
        if 'epochs' in self.graph_specs:
            return int(self.graph_specs['epochs'])
        return 100

    def batch_size(self):
        """
        Return batch size, each dataset has own batch size.
        Model batch size
        :return:
        """
        if 'batch_size' in self.graph_specs:
            return int(self.graph_specs['batch_size'])
        return 32

    def num_layers(self):
        """

        :return:
        """
        if 'num_layers' in self.graph_specs:
            return int(self.graph_specs['num_layers'])
        return 4

    def parameter_shrink(self):
        """

        """
        return self.graph_specs['parameter_shrink']

    def test_batch_size(self):
        """

        """
        return self.graph_specs['test_batch_size']

    def test_total_size(self):
        """

        """
        return self.graph_specs['test_total_size']

    def lr(self) -> float:
        """
        Models default learning rate, default 0.003
        """
        if 'lr' in self.graph_specs:
            float(self.graph_specs['lr'])

        return 0.003

    def set_lr(self, rate: float):
        """
        Adjust learning rate for model, during loading learning rate re-adjusted.
        """
        self.graph_specs['ls'] = rate

    def lr_rate(self):
        """

        """
        return float(self.graph_specs['lr_rate'])

    def set_lr_rate(self, rate: float):
        """

        :param rate:
        :return:
        """
        self.graph_specs['lr_rate'] = rate

    def milestones(self):
        """

        :return:
        """
        return self.graph_specs['milestones']

    def set_milestones(self, m):
        """

        :param m:
        :return:
        """
        self.graph_specs['milestones'] = m

    def load(self) -> bool:
        """
        Return true if model must be loaded.
        """
        return bool(self.config['load_model'])

    def save(self) -> bool:
        """
        Return true if model saved during training.
        """
        return bool(self.config['save_model'])

    def load_epoch(self) -> int:
        """
        Setting dictates whether load model or not.
        """
        return int(self.config['load_epoch'])

    def do_bfs(self) -> bool:
        """
        In Graph RNN uses BFS to predict edges for generated nodes.
        Default: True
        """
        if 'do_bfs' in self.graph_specs:
            return self.graph_specs['do_bfs']
        return True

    def do_randomwalk(self) -> bool:
        """
        Alternative approach is to sample random walks,
        Default: False
        """
        if 'do_randomwalk' in self.graph_specs:
            return self.graph_specs['do_randomwalk']
        return False

    def max_nodes(self) -> int:
        """
        Maximum nodes for a graph generation.
        @return:
        """
        if 'max_num_node' in self.graph_specs:
            return int(self.graph_specs['max_num_node'])
        else:
            self.graph_specs['max_num_node'] = int(0)

        return 0

    def max_depth(self) -> int:
        """
        Maximum nodes to track in BFS.
        @return: Maximum nodes to track in BFS or zero
        """
        if 'max_prev_node' in self.graph_specs:
            return int(self.graph_specs['max_prev_node'])
        else:
            self.graph_specs['max_prev_node'] = int(0)

        return 0

    def set_max_num_node(self, m: int):
        """

        @param m:
        @return:
        """
        self.graph_specs['max_num_node'] = m

    def set_max_prev_node(self, m: int):
        """

        @param m:
        @return:
        """
        self.graph_specs['max_prev_node'] = m

    def epochs_log(self) -> int:
        """

        :return:
        """
        """
        Setting dictates when to log each epoch statistic.
        @return:  Default 100
        """
        if self.inited is False:
            raise Exception("Training must be initialized first")

        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'epochs_log' in self._setting:
            return int(self._setting['epochs_log'])

        return 100

    def start_test(self) -> int:
        """
        Setting dictates when to start sampling at training loop epoch.
        @return:  Default 100
        """
        if 'start_test' in self._setting:
            return int(self._setting['start_test'])
        return 100

    def epochs_test(self) -> int:
        """
        Setting dictates when to start predicting at training loop.
        @return:  Default 100
        @return:
        """
        if 'epochs_test' in self._setting:
            return int(self._setting['epochs_test'])
        return 100

    def epochs_save(self) -> int:
        """
        Save model at epochs ,  by default trainer will use early stopping
        TODO add early stopping optional
        """
        if 'epochs_save' in self._setting:
            return int(self._setting['epochs_save'])
        return 100

    def single_shoot(self) -> bool:
        """
        Draw single shoot
        """
        if 'single_shoot' in self.config:
            return bool(self.config['single_shoot'])
        return False

    def trace_prediction_timer(self) -> bool:
        """
        Trace execution timer for prediction
        """
        if 'trace_prediction_timer' in self.config:
            return bool(self.config['trace_prediction_timer'])
        return False

    def trace_training_timer(self) -> bool:
        """
        Trace training timer
        """
        if 'trace_training_timer' in self.config:
            return bool(self.config['trace_training_timer'])
        return False

    def trace_epocs(self) -> int:
        """
        Trace each epocs
        """
        if 'trace_epocs' in self.config:
            return int(self.config['trace_epocs'])
        return 0

    def num_workers(self) -> int:
        """
        Number of worker node used to fetch data from dataset
        """
        if 'training' in self.config:
            t = self.config['training']
            if 'sample_time' in t:
                return int(t['num_workers'])
        return 0

    def batch_ratio(self) -> int:
        """
        Number batches of samples per each epoch, 1 epoch = n batches
        """
        if self._batch_ratio is not None:
            return self._batch_ratio

        if 'training' in self.config:
            t = self.config['training']
            if 'sample_time' in t:
                self._batch_ratio = int(t['batch_ratio'])
                return self._batch_ratio

        self._batch_ratio = 32

        return self._batch_ratio

    def setup_tensorflow(self):
        """
        Setup tensorflow dir
        """
        time = strftime("%Y-%m-%d-%H", gmtime())
        fmt_print("tensorboard log dir", self.log_dir())
        logging.basicConfig(filename=str(self.dir_log / Path('train' + time + '.log')), level=logging.DEBUG)
        if bool(self.config['regenerate']):
            if os.path.isdir("tensorboard"):
                shutil.rmtree("tensorboard")
        self.writer = SummaryWriter()
        return self.writer

    def build_dir(self):
        """
        Creates all directories required for trainer.
        """
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.isdir(self.dir_log):
            os.makedirs(self.dir_log)

        if not os.path.isdir(self.dir_graph_save):
            os.makedirs(self.dir_graph_save)

        if not os.path.isdir(self.dir_figure):
            os.makedirs(self.dir_figure)

        if not os.path.isdir(self.dir_timing):
            os.makedirs(self.dir_timing)

        if not os.path.isdir(self.dir_model_prediction):
            os.makedirs(self.dir_model_prediction)

        # if not os.path.isdir(self.nll_save_path):
        #     os.makedirs(self.nll_save_path)

    def log_tensorboard(self, loss, epoch, batch_idx):
        """
         Write tensorboard logs

        @param loss:
        @param epoch:
        @param batch_idx:
        @return:
        """
        self.writer.add_scalar('loss_' + self.active_model, loss.item(), epoch * self.batch_ratio() + batch_idx)
        self.writer.flush()

    def log_prediction(self, model, graps):
        """

        """
        # graph_list = utils.load_graph_prediction(graps)
        #
        # # grid = torchvision.utils.make_grid(images_buffer_generator(graph_list))
        # # self.writer.add_image("images", grid)
        # for img in images_buffer_generator(graph_list):
        #     self.writer.add_image(model, img)
        # #       self.writer.add_graph(model, images_buffer_generator(graph_list))

        # self.writer.close()

    def root_dir(self):
        """
        Return root dir where results, dataset stored.
        By default it same dir where we execute code.
        """
        return self.config['root_dir']

    def results_dir(self) -> str:
        """
        Return main directory where all results stored.
        """
        if 'results_dir' in self.config:
            return self.config['results_dir']
        return 'results'

    def log_dir(self) -> str:
        """
        Return directory that used to store logs.
        """
        if 'log_dir' in self.config:
            return self.config['log_dir']
        return 'logs'

    def graph_dir(self) -> str:
        """
        Return directory where store original graphs
        """
        if 'graph_dir' in self.config:
            return self.config['graph_dir']
        return 'graphs'

    def timing_dir(self) -> str:
        """
        Return directory we use to store time traces
        """
        if 'timing_dir' in self.config:
            return self.config['timing_dir']
        return 'timing'

    def model_save_dir(self) -> str:
        """
        Default dir where model checkpoint stored.
        """
        if 'model_save_dir' in self.config:
            return self.config['model_save_dir']
        return 'model_save'

    def prediction_dir(self) -> str:
        """
        Default dir where model prediction serialized.
        """
        if 'figures_prediction_dir' in self.config:
            return self.config['figures_prediction_dir']

        return 'prediction'

    def prediction_figure_dir(self) -> str:
        """
        Default dir where model prediction serialized.
        """
        if 'figures_prediction_dir' in self.config:
            return self.config['prediction_figures']

        return 'prediction'

    def figures_dir(self) -> str:
        """
        Default dir where test figures serialized.
        """
        if 'figures_dir' in self.config:
            return self.config['figures_dir']
        return 'figures'

    def test_ratio(self) -> float:
        """
        Test ratio for test/validation
        """
        if 'training' in self.config:
            t = self.config['training']
            if 'test_ration' in t:
                return float(t['test_ration'])
        return 0.0

    def train_ratio(self) -> float:
        """
         Train/Test ratio.
        """
        if 'training' in self.config:
            t = self.config['training']
            if 'train_ratio' in t:
                return float(t['train_ratio'])
        return 0.0

    def validation_ratio(self) -> float:
        """
        Validation ratio for test/validation
        """
        if 'training' in self.config:
            t = self.config['training']
            if 'validation_ratio' in t:
                return float(t['validation_ratio'])
        return 0.0

    def is_train_network(self) -> bool:
        """
        Train network or not.
        Default: True
        """
        if 'train' in self.config:
            return bool(self.config['train'])
        return True

    def is_draw_samples(self) -> bool:
        """
        Draw sample or not.
        Default: True
        """
        if 'draw_prediction' in self.config:
            return bool(self.config['draw_prediction'])
        return True

    def betas(self, alias_name, default=False) -> [float, float]:
        """
         Coefficients used for computing running averages of gradient
         and its square (default: (0.9, 0.999))

        :param alias_name:
        :param default:
        :return:
        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'betas' in opt:
                return opt['betas']
        return [0.9, 0.999]

    def eps(self, alias_name, default=False):
        """
        Term added to the denominator to improve numerical stability
        default: 1e-8
        :param alias_name:
        :param default:
        :return:
        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'eps' in opt:
                return float(opt['eps'])
        return 1e-8

    def weight_decay(self, alias_name: str, default=False) -> float:
        """

        :param alias_name:
        :param default:
        :return:
        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'weight_decay' in opt:
                return float(opt['weight_decay'])
        return float(0)

    def optimizer_type(self, alias_name: str, default=False) -> str:
        """
        Return optimizer type for a given alias , if default is passed , will return default.

        :param alias_name:  alias_name/
        :param default: if default value needed
        :return: optimizer type, Default Adam
        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'type' in opt:
                return str(opt['type'])

        return "Adam"

    def amsgrad(self, alias_name: str, default=False) -> bool:
        """
         Setting dictates whether to use the AMSGrad variant.

        :param alias_name:
        :param default:
        :return: true if ams grad must be enabled, default False

        """
        if default is False:
            opt = self.get_optimizer(alias_name)
            if 'amsgrad' in opt:
                return bool(opt['amsgrad'])

        return False

    def sample_time(self) -> int:
        """
        :return: number of time take sample during prediction.
        """
        if 'training' in self.config:
            t = self.config['training']
            if 'sample_time' in t:
                return int(t['sample_time'])
        return 0

    def momentum(self, alias_name) -> float:
        """
        Moment factor, default 0
        @param alias_name:
        @return: moment factor, default 0
        """
        opt = self.get_optimizer(alias_name)
        if 'momentum' in opt:
            return float(opt['momentum'])

        return float(0)

    def dampening(self, alias_name) -> float:
        """
        :return: Dampening for momentum, default 0
        """
        opt = self.get_optimizer(alias_name)
        if 'dampening' in opt:
            return float(opt['dampening'])
        return float(0)

    def nesterov(self, alias_name) -> bool:
        """
        Nesterov momentum,
        Default False
        """
        opt = self.get_optimizer(alias_name)
        if 'nesterov' in opt:
            return bool(opt['nesterov'])

        return False

    def get_model_lr_scheduler(self, model_name) -> str:
        """

        :param model_name:
        :return:
        """
        return self.model[model_name]['lr_scheduler']

    def get_model_optimizer(self, model_name) -> str:
        """
        Return model optimizer alias name
        :param model_name:
        :return:
        """
        return self.model[model_name]['optimizer']

    def lr_scheduler(self, alias_name):
        """
        Returns lr scheduler by name, each value of lr_scheduler in dict
        :param alias_name: alias name defined in config.
                           The purpose of alias bind different scheduler config to model
        :return:
        """
        if self.lr_schedulers is not None:
            for elm in self.lr_schedulers:
                spec = elm
                if 'name' in spec and alias_name in spec['name']:
                    return spec
        return None

    def lr_scheduler_type(self, alias_name):
        """
        Returns lr scheduler type.
        :param alias_name: alias_name: alias name defined in config.
                           The purpose of alias bind different scheduler config to model
        :return:
        """
        scheduler_spec = self.lr_scheduler(alias_name)
        if scheduler_spec is not None:
            if 'type' in scheduler_spec:
                return scheduler_spec['type']

        return None

    def min_lr(self, name):
        pass

    def compute_num_samples(self):
        """

        """
        return self.batch_size() * self.batch_ratio()

    def lr_lambdas(self, alias_name):
        """
        TODO
        @param alias_name:
        @return:
        """
        pass

    def is_read_benchmark(self):
        """
        TODO
        @return:
        """
        if 'debug' in self.config:
            t = self.config['debug']
            if 'benchmark_read' in t:
                return bool(t['benchmark_read'])
        return False

    def is_graph_creator_verbose(self) -> bool:
        """
        Flag dictates if need trace debug on train graph generation process.
        Default false
        @return:
        """
        if 'debug' in self.config:
            t = self.config['debug']
            if 'graph_generator' in t:
                return bool(t['graph_generator'])
        return False

    def is_model_verbose(self):
        """
        Enables model debug during creation
        @return:
        """
        if 'debug' in self.config:
            t = self.config['debug']
            if 'model_creation' in t:
                return bool(t['model_creation'])
        return False

    def is_train_verbose(self):
        """

        @return:
        """
        if 'debug' in self.config:
            t = self.config['debug']
            if 'train_verbose' in t:
                return bool(t['train_verbose'])
        return False

    def get_active_train_graph(self):
        """

        """
        return self.active

    def get_active_train_graph_spec(self):
        """

        """
        return self.graph_specs

    def get_active_model_prediction_files(self, reverse=True) -> List[str]:
        """
         Method return all prediction model generated.
        """
        if not self.is_trained():
            raise Exception("Untrained model")

        if not os.path.exists(self.get_prediction_dir()):
            return []

        only_files = [f for f in listdir(self.get_prediction_dir())
                      if isfile(join(self.get_prediction_dir(), f)) and
                      f.find(self.get_active_model()) != -1 and
                      f.find(self.get_active_train_graph()) != -1]

        # compute position of 'epoch' at runtime
        if len(only_files) > 0:
            pos = only_files[0].find(self._epoch_file_fmt)
            if pos != -1:
                pos = pos + len(self._epoch_file_fmt) + 1

        def epoch_sort(x):
            """
            Sort file by epoch.
            """
            file_suffix = x[pos:]
            if len(file_suffix) > 0:
                tokens = file_suffix.split('_')
                if tokens[0].isnumeric():
                    return int(tokens[0])

            return 0

        only_files.sort(key=epoch_sort, reverse=reverse)
        return only_files

    def get_active_model(self) -> str:
        """
         Return model that indicate as current active model.
         It important to understand, we can switch between models.
        """
        return self.active_model

    def get_active_model_spec(self):
        """
        Return model specs.  Each value is dict.
        """
        return self.model

    def get_active_model_names(self):
        """
        Return model specs.  Each value is dict.
        """
        return self.model.keys()

    def get_last_graph_predictions(self):
        self.get_active_model_prediction_files()

    # def compute_basic_stats(real_g_list, target_g_list):
    #     dist_degree = eval.stats.degree_stats(real_g_list, target_g_list)
    #     dist_clustering = eval.stats.clustering_stats(real_g_list, target_g_list)
    #     return dist_degree, dist_clustering

    def get_last_graph_stat(self, num_samples=1) -> List[nx.classes.graph.Graph]:
        """
        Method return last graph from generated graph list.
        """
        files = self.get_active_model_prediction_files()
        last_file = files[0]
        last_file_path = self.get_prediction_dir() / Path(last_file)
        graphs = graph_from_file(last_file_path)
        return graphs

    def get_prediction_graph(self, num_samples=1, reverse=False, is_last=False) -> List[nx.classes.graph.Graph]:
        """
        Method return generator for all prediction files.
        A Caller can iterate each iter call will return one file name.

        Note file are sorted.
        :param num_samples:
        :param reverse:
        :param is_last:
        :return:
        """
        if is_last is True:
            files = self.get_active_model_prediction_files(reverse=reverse)
            epoch = extract_value_from_file(files, "epoch")
            sample_time = extract_value_from_file(files, "sample")
            graph_file = self.get_prediction_dir() / Path(files)
            yield epoch, sample_time, graph_file, graph_from_file(graph_file)

        files = self.get_active_model_prediction_files(reverse=reverse)
        for f in files:
            epoch = extract_value_from_file(f, "epoch")
            sample_time = extract_value_from_file(f, "sample")
            graph_file = self.get_prediction_dir() / Path(f)
            print(graph_file)
            yield epoch, sample_time, graph_file, graph_from_file(graph_file)

    def is_trained(self) -> bool:
        """
        Return true if model trainer,  it mainly checks if dat file created or not.
        :return: True if trainer
        """
        models_filenames = self.model_filenames()
        if self._verbose:
            print("Model filenames", models_filenames)

        for k in models_filenames:
            if not os.path.isfile(models_filenames[k]):
                return False

        return True

    def get_last_saved_epoc(self):
        """
         Return last checkpoint saved as dict where key sub-model: last checkpoint
         If model un trained will raise exception
        """
        checkpoints = {}
        if self._verbose:
            fmtl_print('Trying load models last checkpoint...', self.active_model)

        if not self.is_trained():
            raise Exception("Untrained model")

        models_filenames = self.model_filenames()
        if self._verbose:
            print("Model filenames", models_filenames)

        for m in models_filenames:
            if self._verbose:
                print("Trying to load checkpoint file", models_filenames[m])

            check = torch.load(models_filenames[m])
            if self._verbose:
                print(check.keys())

            if 'epoch' in check:
                checkpoints[m] = check['epoch']

        return checkpoints

    # def load_models(self, model):
    # """
    # TODO not important for noe
    # """
    # checkpoints = {}
    # models_filenames = self.model_filenames()
    # print('Trying load models last checkpoint...')
    # models_filenames = self.model_filenames()
    # print("Model filenames", models_filenames)
    # for m in models_filenames:
    #         checkpoints[m] = torch.load(models_filenames[m])
    #         model.state.rnn.load_state_dict(checkpoint['model_state_dict'])
    #         model.optimizer_rnn.load_state_dict(checkpoints[m]['optimizer_state_dict'])
    #         model.scheduler_rnn.load_state_dict(checkpoints[m]['optimizer_state_dict'])

    # return checkpoints
    def generate_prediction_figure_name(self, epoch, gid=None, sample_time=1, file_type='png'):
        """

        """
        return self.prediction_filename(epoch, sample=sample_time, gid=gid, file_type=file_type)

    def load_train_test(self):
        """
        """
        return graph_from_file(self.train_graph_file()), graph_from_file(self.train_graph_file())

    def is_evaluate(self) -> bool:
        """
        TODO
        """
        if 'evaluate' in self.config:
            return bool(self.config['evaluate'])

        return False

    #
    # def prediction_files_generator(self):
    #     files = self.get_active_model_prediction_files()
    #     yield iter(files)

    def done(self):
        self.writer.close()

    def get_model_submodels(self) -> List[str]:
        keys = self.model.keys()
        return [k for k in keys]

    def get_optimizer_type(self, optimizer_alias_name):
        """
        Method return optimizer type
        :param optimizer_alias_name:
        :return:
        """
        opt = self._optimizers[optimizer_alias_name]
        if 'type' in opt:
            return opt['type']
        raise Exception("Optimizer has no type defined")

    def is_early_stopping(self) -> bool:
        """
        Return true if early stopping enabled.
        :return:  default value False
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'early_stopping' in self._setting:
            return True

        return False

    def get_patience(self) -> int:
        """
        Number of epochs with no improvement after which training will be stopped.
        :return: default value False
        """
        if 'early_stopping' in self.config:
            stopping = self.config['early_stopping']
            if 'patience' in stopping:
                return int(stopping['patience'])

        return False

    def mmd_degree(self) -> bool:
        """

        :return:
        """
        if 'metrics' in self.config:
            metrics = self.config['metrics']
            if 'degree' in metrics:
                return metrics['degree']
        return False

    def is_log_lr_rate(self):
        """
        TODO
        :return:
        """
        True

    def mmd_clustering(self) -> bool:
        """
        :return:
        """
        return True

    def mmd_orbits(self) -> bool:
        """
        :return:
        """
        return True

    def tensorboard_sample_update(self):
        """
        Return true if early stopping enabled.
        :return:  default value False
        """
        if self._setting is None:
            raise Exception("Initialize settings first")

        if 'early_stopping' in self._setting:
            return True

    def set_depth(self, graph_depth):
        """

        @param graph_depth:
        @return:
        """
        if 'max_prev_node' in self.graph_specs:
            self.graph_specs['max_prev_node'] = graph_depth
