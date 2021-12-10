import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable


def binary_cross_entropy_weight(y_pred, y, has_weight=False, weight_length=1, weight_max=10, device='cuda'):
    """
    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_max: the magnitude that the weight is enhanced
    :return:

    @param device:
    @param weight_max:
    @param weight_length:
    @param y_pred:
    @param has_weight:
    @return: weighted cross entropy
    """
    if has_weight:
        # compute weight
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


def sample_sigmoid(y, sample, thresh=0.5, device='cuda'):
    """
    Perform sigmoid on y and sampling over un normalized value.

    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param device: device cpu or cuda
    :return: sampled result
    """

    y = torch.sigmoid(y)
    if not sample:
        y_thresh = Variable(torch.ones(y.size(0),
                                       y.size(1),
                                       y.size(2)) * thresh).to(device)
        y_result = torch.gt(y, y_thresh).float()
        return y_result

    y_thresh = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
    y_result = torch.gt(y, y_thresh).float()
    return y_result


def sample_sigmoid_weighted(y, sample, thresh=0.5, sample_time=2, device='cuda'):
    """
     Perform sigmoid on y and sampling over un normalized value.

    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sample_time: how many times do we sample, if =1, do single sample
    :param device: device cpu or cuda
    :return: sampled result
    """

    y = torch.sigmoid(y)
    if not sample:
        y_thresh = Variable(torch.ones(y.size(0),
                                       y.size(1),
                                       y.size(2)) * thresh).to(device)
        y_result = torch.gt(y, y_thresh).float().to(device)
        return y_result

    # sample
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
