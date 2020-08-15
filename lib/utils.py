# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import torch
import time
import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class TextLogger(object):
    """Write log immediately to the disk"""
    def __init__(self, filepath):
        self.f = open(filepath, 'w')
        self.fid = self.f.fileno()
        self.filepath = filepath

    def close(self):
        self.f.close()

    def write(self, content):
        self.f.write(content)
        self.f.flush()
        os.fsync(self.fid)

    def write_buf(self, content):
        self.f.write(content)

    def print_and_write(self, content):
        print(content)
        self.write(content+'\n')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    num = output.size(1)
    target_topk = []
    appendices = []
    for k in topk:
        if k <= num:
            target_topk.append(k)
        else:
            appendices.append([0.0])
    topk = target_topk
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res + appendices


def to_numpy(var):
    use_cuda = torch.cuda.is_available()
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()


def to_tensor(ndarray, requires_grad=False):  # return a float tensor by default
    tensor = torch.from_numpy(ndarray).float()  # by default does not require grad
    if requires_grad:
        tensor.requires_grad_()
    return tensor.cuda() if torch.cuda.is_available() else tensor


def measure_layer_for_pruning(layer, x):
    def get_layer_type(layer):
        layer_str = str(layer)
        return layer_str[:layer_str.find('(')].strip()

    def get_layer_param(model):
        import operator
        import functools

        return sum([functools.reduce(operator.mul, i.size(), 1) for i in model.parameters()])

    multi_add = 1
    type_name = get_layer_type(layer)

    # ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        layer.flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        layer.params = get_layer_param(layer)
    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        layer.flops = weight_ops + bias_ops
        layer.params = get_layer_param(layer)
    return


def least_square_sklearn(X, Y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    return reg.coef_


def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    return parent_dir


# logging
def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))