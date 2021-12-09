# Model RNN Trainer class
#
# Responsible for training RNN model such as GraphRNN.
#
# Mustafa
import time
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils as tutil
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.transforms import ToTensor
from tqdm import tqdm, tnrange

from .generator_trainer import GeneratorTrainer
from .graph_tools import graph_from_tensors
from .model_config import ModelSpecs
from .models.early_stoping import EarlyStopping
from .models.AbstractGraphDecoder import AbstractGraphDecoder
from .models.model_utils import sample_sigmoid
from .plotlib.plot import graph2image_buffer
from .utils import get_graph, fmt_print, fmtl_print


class RnnGenerator(GeneratorTrainer):
    """
    Rnn Generator Trainer
    """

    def __init__(self, trainer_spec: ModelSpecs,
                 dataset: tutil.data.DataLoader,
                 models: dict,
                 decoder: AbstractGraphDecoder,
                 device='cpu',
                 verbose=False,
                 is_notebook=True):
        """
        Rnn Generator requires two networks, one for node level prediction and second for edge level prediction.
        """
        super(RnnGenerator, self).__init__(verbose=verbose, is_notebook=is_notebook)
        #
        # self.is_notebook = is_notebook
        fmt_print("Creating generator", "RnnGenerator")

        # this is mostly for debugging code
        self.debug = False

        # set verbose output
        self.verbose = verbose

        # if we run trainer in notebook mode
        self.is_notebook = is_notebook

        # trainer spec
        self.trainer_spec = trainer_spec

        # device that model suppose to be already in
        self.device = device

        # dataset
        self.dataset = dataset

        if 'node_model' not in models:
            raise Exception("model require node_mode.")
        self.node_rnn = models['node_model']

        if 'edge_model' not in models:
            raise Exception("model require edge_model.")
        self.edge_rnn = models['edge_model']

        # both optimizer and scheduler part of model spec.
        # we use alias name to bind different configuration of optimizer
        # and scheduler to a model.
        self.model_spec = trainer_spec.model

        # optimizer alias name
        self.node_rnn_opt_name = self.model_spec['node_model']['optimizer']
        self.edge_rnn_opt_name = self.model_spec['edge_model']['optimizer']

        # same for scheduler get name
        # TODO move this to train spec and wrap around exception handler
        self.lr_node_rnn_name = self.model_spec['node_model']['lr_scheduler']
        self.lr_edge_rnn_name = self.model_spec['edge_model']['lr_scheduler']

        # initialize optimizer
        self.optimizer_node_rnn = self.create_optimizer(self.node_rnn, self.node_rnn_opt_name)
        self.optimizer_edge_rnn = self.create_optimizer(self.edge_rnn, self.edge_rnn_opt_name)

        # scheduler created per each network and can be different spaced in config.yaml
        self.scheduler_rnn = self.create_lr_scheduler(self.optimizer_node_rnn, self.lr_node_rnn_name)
        self.scheduler_edge_rnn = self.create_lr_scheduler(self.optimizer_edge_rnn, self.lr_edge_rnn_name)

        # decoder that will be used to decode node adj
        self.decoder = decoder

        # All weights are equal to 1
        pos_weight = torch.ones([1]).to(self.device)
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, size_average=True, reduction='mean')
        if 'bce' in trainer_spec.model:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = None

        if verbose:
            print(trainer_spec.model)

    def __call__(self):
        raise Exception("unsupported")

    def create_optimizer(self, net, alias_name: str):
        """
        Create Optimizer based on trainer specs
        @param net: network that we used to attach optimizer
        @param alias_name: alias name that used to bind to network. model spec
                           has alias name that used to bind optimizer to a network
        @return: optimizer
        """
        optimizer_type = self.trainer_spec.optimizer_type(alias_name)

        if optimizer_type == 'Adam':
            if self.verbose:
                fmtl_print("Creating {} type optimizer.".format(alias_name), "Adam")
                fmtl_print("Creating {} lr".format(alias_name), self.trainer_spec.lr())
                fmtl_print("Creating {} betas".format(alias_name), self.trainer_spec.betas(alias_name))
                fmtl_print("Creating {} eps".format(alias_name), self.trainer_spec.eps(alias_name))
                fmtl_print("Creating {} weight decay".format(alias_name), self.trainer_spec.weight_decay(alias_name))
                fmtl_print("Creating {} amsgrad".format(alias_name), self.trainer_spec.amsgrad(alias_name))

            opt = optim.Adam(list(net.parameters()),
                             lr=self.trainer_spec.lr(),
                             betas=self.trainer_spec.betas(alias_name),
                             eps=self.trainer_spec.eps(alias_name),
                             weight_decay=self.trainer_spec.weight_decay(alias_name),
                             amsgrad=self.trainer_spec.amsgrad(alias_name))
        elif optimizer_type == 'SGD':
            if self.debug:
                fmtl_print("Creating {} optimizer.".format(alias_name), "SGD")
            opt = optim.opt = optim.SGD(list(net.parameters(alias_name)),
                                        momentum=self.trainer_spec.momentum(alias_name),
                                        dampening=self.trainer_spec.dampening(alias_name),
                                        weight_decay=self.trainer_spec.weight_decay(alias_name),
                                        nesterov=self.trainer_spec.nesterov(alias_name))
        elif self.trainer_spec.optimizer_type == 'none':
            opt = None
        else:
            raise ValueError("unknown optimizer: {}".format(optimizer_type))

        return opt

    def create_lr_scheduler(self, optimizer, alias_name: str):
        """
        Creates lr scheduler based on specs
        @param optimizer:
        @param alias_name:
        @return:
        """
        lr_scheduler_type = self.trainer_spec.lr_scheduler_type(alias_name)
        if lr_scheduler_type == 'cos':
            if self.verbose:
                fmtl_print("Creating {} lr scheduler.".format(alias_name), "cos")
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=self.trainer_spec.epochs(),
                                                       eta_min=self.trainer_spec.min_lr())
        elif lr_scheduler_type == 'multistep':
            if self.verbose:
                fmtl_print("Creating {} lr scheduler.".format(alias_name), "multistep")
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=self.trainer_spec.milestones(),
                                                 gamma=self.trainer_spec.lr_rate())
        elif lr_scheduler_type == 'exp-warmup':
            if self.verbose:
                fmtl_print("Creating {} lr_scheduler_type.".format(alias_name), "exp-warmup")
            lr_lambdas = self.trainer_spec.lr_lambdas(alias_name)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        elif lr_scheduler_type == 'none':
            if self.verbose:
                fmtl_print("Creating {} optimizer.".format(alias_name), "none")
            scheduler = None
        else:
            raise ValueError("unknown scheduler: {}".format(lr_scheduler_type))

        return scheduler

    @torch.no_grad()
    def model_prediction(self):
        """
        Model Prediction,  Generate graph prediction
        and saves each prediction as graph file.
        :return:
        """
        batch_size = self.trainer_spec.test_batch_size()
        self.node_rnn.hidden = self.node_rnn.init_hidden(batch_size).to(self.device)
        self.node_rnn.eval()
        self.edge_rnn.eval()

        # generate graphs
        max_num_node = int(self.trainer_spec.max_nodes())

        # prediction
        y_final_prediction = Variable(torch.zeros(batch_size,
                                                  max_num_node,
                                                  self.trainer_spec.max_depth())).to(self.device)

        x_step = Variable(torch.ones(batch_size, 1,
                                     self.trainer_spec.max_depth()), requires_grad=False).to(self.device)

        for i in range(max_num_node):
            h = self.node_rnn(x_step)
            hidden_null = Variable(torch.zeros(self.trainer_spec.num_layers() - 1,
                                               h.size(0), h.size(2))).to(self.device)

            # num_layers, batch_size, hidden_size
            self.edge_rnn.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)
            x_step = Variable(torch.zeros(batch_size, 1, self.trainer_spec.max_depth())).to(self.device)
            output_x_step = Variable(torch.ones(batch_size, 1, 1)).to(self.device)
            for j in range(min(self.trainer_spec.max_depth(), i + 1)):
                # get prediction for edge
                output_y_pred_step = self.edge_rnn(output_x_step)
                output_x_step = sample_sigmoid(output_y_pred_step,
                                               sample=True,
                                               sample_time=1,
                                               device=self.device)
                # adjust edge
                x_step[:, :, j:j + 1] = output_x_step
                self.edge_rnn.hidden = Variable(self.edge_rnn.hidden.data).to(self.device)
            y_final_prediction[:, i:i + 1, :] = x_step
            self.node_rnn.hidden = Variable(self.node_rnn.hidden.data).to(self.device)

        y_pred_long_data = y_final_prediction.data.long()

        # save graphs
        pred_list = []
        for i in range(batch_size):
            adj_pred = self.decoder.decode(y_pred_long_data[i].cpu().numpy())
            # get a graph from zero-padded adj
            pred_list.append(get_graph(adj_pred))

        return pred_list

    def monitor(self, epoch, loss):
        print(
            'Epoch: {}/{}, train loss: {:.6f}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, self.trainer_spec.epochs(), loss.item(), loss.item(),
                self.trainer_spec.graph_type,
                self.trainer_spec.num_layers(),
                self.trainer_spec.hidden_size_rnn))

    def train_epoch(self, epoch):
        """

        :param epoch:
        :return:
        """

        self.node_rnn.train()
        self.edge_rnn.train()
        twriter = self.trainer_spec.writer

        total_loss = 0.0
        for batch_idx, data in enumerate(self.dataset):

            self.node_rnn.zero_grad()
            self.edge_rnn.zero_grad()

            y_len_unsorted = data['len']
            y_len_max = max(y_len_unsorted)
            x_unsorted = data['x'][:, 0:y_len_max, :].float()
            y_unsorted = data['y'][:, 0:y_len_max, :].float()

            # init according to batch size
            self.node_rnn.hidden = self.node_rnn.init_hidden(batch_size=x_unsorted.size(0)).to(self.device)

            # sort input
            y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
            y_len = y_len.numpy().tolist()
            x = torch.index_select(x_unsorted, 0, sort_index)
            y = torch.index_select(y_unsorted, 0, sort_index)

            # input, output for output rnn module
            y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
            y_idx = torch.flip(torch.arange(y_reshape.size(0), requires_grad=False), [0])

            y_reshape = y_reshape.index_select(0, y_idx)
            y_reshape = y_reshape.view(y_reshape.size(0),
                                       y_reshape.size(1), 1)

            output_x = torch.cat((torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1)
            output_y = y_reshape

            # batch size for output module: sum(y_len)
            output_y_len = []
            output_y_len_bin = np.bincount(np.array(y_len))

            # loop_time = time.time()
            for i in range(len(output_y_len_bin) - 1, 0, -1):
                # count how many y_len is above i
                count_temp = np.sum(output_y_len_bin[i:])
                # put them in output_y_len; max value should not exceed y.size(2)
                output_y_len.extend([min(i, y.size(2))] * count_temp)

            # pack into variable
            x = Variable(x).to(self.device)
            y = Variable(y).to(self.device)
            output_x = Variable(output_x).to(self.device)
            output_y = Variable(output_y).to(self.device)

            # if using ground truth to train
            h = self.node_rnn(x, pack=True, input_len=y_len).to(self.device)
            # get packed hidden vector
            h = pack_padded_sequence(h, y_len, batch_first=True).to(self.device).data

            indices = torch.flip(torch.arange(h.size(0), requires_grad=False), [0]).to(self.device)
            h = h.index_select(0, indices)
            hidden_null = Variable(torch.zeros(self.trainer_spec.num_layers() - 1,
                                               h.size(0), h.size(1)), requires_grad=False).to(self.device)
            # num_layers, batch_size, hidden_size
            self.edge_rnn.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
            y_predicted = self.edge_rnn(output_x, pack=True, input_len=output_y_len)
            if self.criterion is None:
                y_predicted = torch.sigmoid(y_predicted)

            # pack and pad
            y_predicted = pack_padded_sequence(y_predicted, output_y_len, batch_first=True)
            y_predicted = pad_packed_sequence(y_predicted, batch_first=True)[0]
            output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
            output_y = pad_packed_sequence(output_y, batch_first=True)[0]

            # use cross entropy loss
            # loss = F.binary_cross_entropy(y_pred, output_y)
            if self.criterion is None:
                loss = F.binary_cross_entropy(y_predicted, output_y)
            else:
                loss = self.criterion(y_predicted, output_y)

            loss.backward()

            # update both optimizers
            self.optimizer_edge_rnn.step()
            self.optimizer_node_rnn.step()

            # run lr_scheduler
            if self.scheduler_edge_rnn is not None:
                self.scheduler_edge_rnn.step()
            if self.scheduler_rnn is not None:
                self.scheduler_rnn.step()

            # correct += (output == labels).float().sum()
            # only output first batch's statistics
            if epoch > 0 and epoch % self.trainer_spec.epochs_log() == 0 and batch_idx == 0:
                print(
                    'Epoch: {}/{}, train loss: {:.6f}, train loss: {:.6f}, num_layer: {}, hidden: {}'.format(
                        epoch, self.trainer_spec.epochs(), loss.item(), loss.item(),
                        self.trainer_spec.num_layers(),
                        self.trainer_spec.hidden_size_rnn))

            # print(y_pred.sum())
            # print(output_y.sum())
            # correct += (y_pred == output_y).sum().item()
            # total += y_pred.size(0)
            #
            # feature_dim = y.size(1) * y.size(2)
            # loss_sum += loss.item() * feature_dim
            total_loss += loss.item()
            # self.trainer_spec.log_tensorboard(loss, epoch, batch_idx)
            twriter.add_scalar('loss_' + self.trainer_spec.active_model, loss.item(),
                                epoch * self.trainer_spec.batch_ratio() + batch_idx)

        total_loss /= batch_idx + 1
        twriter.add_scalar('total_loss_' + self.trainer_spec.active_model, total_loss,
                           epoch * self.trainer_spec.batch_ratio() + batch_idx)
        # model logs
        for name, weight in self.node_rnn.named_parameters():
            twriter.add_histogram(name, weight, epoch)
            twriter.add_histogram(f'{name}.grad', weight.grad, epoch)

        for name, weight in self.edge_rnn.named_parameters():
            twriter.add_histogram(name, weight, epoch)
            twriter.add_histogram(f'{name}.grad', weight.grad, epoch)

        twriter.flush()
        return total_loss

    def load(self):
        """
        Method loads model from a checkpoint.
        :return: last epoch
        """
        if self.trainer_spec.load():
            print('attempting to load {} model node weights state_dict '
                  'loaded from {}...'.format(self.trainer_spec.get_active_model(),
                                             self.trainer_spec.model_node_file_name()))
            # load trained optimizer state_dict
            try:
                checkpoint = torch.load(self.trainer_spec.model_node_file_name())
                if 'model_state_dict' not in checkpoint:
                    raise Exception("model has no state dict")

                self.node_rnn.load_state_dict(checkpoint['model_state_dict'])
                if 'model_state_dict' not in checkpoint:
                    raise Exception("model has no state dict")

                self.optimizer_node_rnn.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'optimizer_state_dict' not in checkpoint:
                    raise Exception("model has no optimizer_state_dict")

                self.scheduler_rnn.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scheduler_state_dict' not in checkpoint:
                    raise Exception("model has no scheduler_state_dict")

                print('attempting to load {} model edge weights state_dict '
                      'loaded...'.format(self.trainer_spec.get_active_model()))

                checkpoint_edge_rnn = torch.load(self.trainer_spec.model_edge_file_name())
                self.edge_rnn.load_state_dict(checkpoint_edge_rnn['model_state_dict'])
                if 'model_state_dict' not in checkpoint:
                    raise Exception("model has no model_state_dict")

                self.optimizer_edge_rnn.load_state_dict(checkpoint_edge_rnn['optimizer_state_dict'])
                if 'optimizer_state_dict' not in checkpoint:
                    raise Exception("model has no optimizer_state_dict")

                self.scheduler_edge_rnn.load_state_dict(checkpoint_edge_rnn['scheduler_state_dict'])

                self.trainer_spec.set_lr(0.00001)
                print('Model loaded, lr: {}'.format(self.trainer_spec.lr()))
                print("Last checkpoint. ", checkpoint['epoch'])

                return checkpoint['epoch']

            except FileNotFoundError as e:
                print("Failed load model files. No saved model found.")

        return 0

    def save_ifneed(self, epoch, last_epoch=False):
        """
         Saves model checkpoint, when based on template settings.

        :param last_epoch: if it last_epoch we always save.
        :param epoch: current epoch
        :return: True if saved
        """
        if self.trainer_spec.save() or last_epoch is True:
            if epoch % self.trainer_spec.epochs_save() == 0:
                if self.trainer_spec.is_train_verbose():
                    fmt_print('Saving node model {}'.format(self.trainer_spec.model_node_file_name()))
                    fmt_print('Saving edge model {}'.format(self.trainer_spec.model_edge_file_name()))

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.node_rnn.state_dict(),
                    'optimizer_state_dict': self.optimizer_node_rnn.state_dict(),
                    'scheduler_state_dict': self.scheduler_rnn.state_dict(),
                }, self.trainer_spec.model_node_file_name())

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.edge_rnn.state_dict(),
                    'optimizer_state_dict': self.optimizer_edge_rnn.state_dict(),
                    'scheduler_state_dict': self.scheduler_edge_rnn.state_dict(),
                }, self.trainer_spec.model_edge_file_name())

                return True

        return False

    def test_prediction_file(self, epoch, sample_time):
        """
        Generated test prediction file name.
        @param epoch: current epoch
        @param sample_time: and from what sample num
        @return:
        """
        return self.trainer_spec.prediction_filename(epoch, sample_time)

    @torch.no_grad()
    def sample(self, epoch, low_bound=1, upper_bound=4):
        """

        Take samples from prediction and saves graph to a file.

        @param epoch:
        @param low_bound:
        @param upper_bound:
        @return:
        """
        for sample_time in range(low_bound, upper_bound):
            predictions = []
            while len(predictions) < self.trainer_spec.test_total_size():
                # num graphs
                predictions.extend(self.model_prediction())

            # plot graph
            if epoch % self.trainer_spec.tensorboard_sample_update() == 0:
                self.plot_example(epoch, predictions[-1])

            # save graph
            self.save_graphs(predictions, self.test_prediction_file(epoch, sample_time), verbose=self.verbose)
            if self.trainer_spec.single_shoot():
                break

    def get_node_rnn_lr(self):
        for param_group in self.node_rnn.param_groups:
            return param_group['lr']

    def get_edge_rnn_lr(self):
        for param_group in self.edge_rnn.param_groups:
            return param_group['lr']

    def plot_example(self, epoch, example=None):
        """

        Plot a graph as image in tensorboard imagaes.

        :param epoch:
        :param example:
        :return:
        """
        twriter = self.trainer_spec.writer
        plot_type = "training"
        A = example

        if example is None:
            graph = next(iter(self.dataset))
            y_len_unsorted = graph['len']
            y_len_max = max(y_len_unsorted)
            y_unsorted = graph['y'][:, 0:y_len_max, :]
            example = y_unsorted[0].cpu().numpy()
            # take y tensor convert to A Matrix
            A = self.decoder.decode(example)
            A = nx.from_numpy_matrix(A)
            plot_type = "prediction"

        A = graph_from_tensors(A)

        # convert to image buffer and torch tensor
        img = graph2image_buffer(A)
        image = ToTensor()(img.copy()).unsqueeze(0)
        images = next(iter(image))
        grid = torchvision.utils.make_grid(images)
        twriter.add_image("images_" + plot_type + self.trainer_spec.active_model, grid, epoch)

    def is_sample_time(self, epoch):
        """
         Default callback when start sample, caller can overwrite this.
        :param epoch:
        :return:
        """
        test_epoch = self.trainer_spec.epochs_test()
        epoch_start = self.trainer_spec.start_test()
        return epoch % test_epoch == 0 and epoch >= epoch_start

    def train(self, is_sample_time=None):
        """

        :return:
        """
        # load last epoch in case we do resuming.
        last_epoch = self.load()
        # tensorboard writer
        twriter = self.trainer_spec.writer
        # early stopping
        early_stopping = None
        if self.trainer_spec.is_early_stopping():
            early_stopping = EarlyStopping(patience=self.trainer_spec.get_patience(), verbose=False)

        # max epoch part of spec
        max_epochs = self.trainer_spec.epochs()
        if self.verbose:
            fmtl_print("max epoch", max_epochs)
            fmtl_print("last epoch", last_epoch)

        active_mode = self.trainer_spec.active_model
        is_sample = self.is_sample_time
        total_epoch_loss = 0.0

        if self.is_notebook:
            tqdm_iter = tnrange(last_epoch, max_epochs)
        else:
            tqdm_iter = tqdm(range(last_epoch, max_epochs))

        tqdm_iter.set_postfix({'total_epoch_loss': 0})

        for epoch in tqdm_iter:

            # self.plot_example(epoch)
            time_start = time.monotonic()
            total_epoch_loss = self.train_epoch(epoch)
            tqdm_iter.set_postfix({'total_epoch_loss': total_epoch_loss})
            twriter.add_scalar('time_trace_' + self.trainer_spec.active_model, time.monotonic() - time_start, epoch)

            # test
            if is_sample(epoch):
                # sample
                if self.trainer_spec.trace_prediction_timer():
                    # prediction_timer = time.time()
                    prediction_timer = time.monotonic()
                self.sample(epoch)
                if self.trainer_spec.trace_prediction_timer() and epoch % self.trainer_spec.trace_epocs() == 0:
                    twriter.add_scalar('sample_time_' + self.trainer_spec.active_model,
                                       time.monotonic() - prediction_timer, epoch)

            if early_stopping is not None:
                tqdm_iter.set_postfix({'total_epoch_loss': total_epoch_loss,
                                       'early stop': early_stopping.counter,
                                       'out of': early_stopping.patience})
                early_stopping(total_epoch_loss, self, epoch=epoch)
                if early_stopping.early_stop:
                    if self.trainer_spec.is_train_verbose():
                        print("Early stopping")
                    break

            if self.trainer_spec.is_log_lr_rate():
                twriter.add_scalar('learning_node_rnn_rate_' + active_mode,
                                   self.scheduler_rnn.get_last_lr()[0], epoch)
                twriter.add_scalar('learning edge_rnn_rate_' + active_mode,
                                   self.scheduler_edge_rnn.get_last_lr()[0], epoch)

            # save model checkpoint
            if self.save_ifneed(epoch):
                tqdm_iter.set_postfix({'total_epoch_loss': total_epoch_loss,
                                       'early stop': early_stopping.counter,
                                       'out of': early_stopping.patience,
                                       'saved': True})
            last_epoch += 1

        self.trainer_spec.done()
        if last_epoch >= max_epochs:
            self.save_ifneed(last_epoch, last_epoch=True)