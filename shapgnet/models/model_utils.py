import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
# from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm
from ..utils import *


def binary_cross_entropy_weight(y_pred, y, has_weight=False, weight_length=1, weight_max=10, device='cpu'):
    """

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    """
    if has_weight:
        weight = torch.ones(y.size(0), y.size(1), y.size(2))
        weight_linear = torch.arange(1, weight_length + 1) / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(y.size(0), 1, y.size(2))
        weight[:, -1 * weight_length:, :] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.to(device))
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def sample_multinomial(y, num_of_samples=1):
    y = F.softmax(y, dim=2)
    sampled_y = torch.mode(torch.multinomial(y.view(y.size(0), y.size(2)),
                                             num_samples=num_of_samples,
                                             replacement=True))[0]
    return sampled_y.reshape(-1, 1)


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2, device='cpu'):
    """
     do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sample_time: how many times do we sample, if =1, do single sample
    :param device: device cpu or cuda
    :return: sampled result
    """

    # do sigmoid first
    y = F.sigmoid(y)
    # do max likelihood based on some threshold
    if not sample:
        y_thresh = Variable(torch.ones(y.size(0),
                                       y.size(1),
                                       y.size(2)) * thresh).to(device)
        y_result = torch.gt(y, y_thresh).float()
        return y_result

    y_thresh = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
    y_result = torch.gt(y, y_thresh).float()

    return y_result


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2, device='cpu'):
    """
     do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sample_time: how many times do we sample, if =1, do single sample
    :param device: device cpu or cuda
    :return: sampled result
    """

    # do sigmoid first
    y = torch.sigmoid(y)
    # do max likelihood based on some threshold
    if not sample:
        y_thresh = Variable(torch.ones(y.size(0),
                                       y.size(1),
                                       y.size(2)) * thresh).to(device)
        y_result = torch.gt(y, y_thresh).float().to(device)
        return y_result

    # do sampling
    if sample_time > 1:
        y_result = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
        # loop over all batches
        for i in range(y_result.size(0)):
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y.size(1), y.size(2))).to(device)
                y_result[i] = torch.gt(y[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    else:
        y_thresh = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
        y_result = torch.gt(y, y_thresh).float()

    return y_result


def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2, device='cpu'):
    """
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    """

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).to(device)
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current < y_len[i]:
            while True:
                y_thresh = Variable(torch.rand(y_pred.size(1),
                                               y_pred.size(2))).to(device)
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                # print('current',current)
                # print('y_result',y_result[i].data)
                # print('y',y[i])
                y_diff = y_result[i].data - y[i]
                if (y_diff >= 0).all():
                    break
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(device)
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result


def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2, device='cpu'):
    """
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    """

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).to(device)
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current < y_len[i]:
            y_result[i] = y[i]
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(device)
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result
