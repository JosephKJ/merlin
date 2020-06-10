import logging
import torch
from lib.config import cfg
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns
import os


def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print(message)


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


class Metrics:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


# one-hot the layer index
def one_hot(data, max_value):
    ones = torch.sparse.torch.eye(max_value)
    return ones.index_select(0, data)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    lr = cfg.kernels.learning_rate * np.minimum((-epoch) * 1. / (cfg.kernels.epochs) + 1, 1.)
    return max(0, lr)


def tonp(x):
    return x.cpu().detach().numpy()


def plot(samples, filename='samples'):
    try:
        d = 7 if '7x7' in cfg.model else 3
        f, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 10))
        for x, ax in zip(samples.reshape((-1, d, d)), axs.flat):
            # sns.heatmap(tonp(x), ax=ax, cmap='Greens')
            ax.axis('off')
        f.savefig(os.path.join(cfg.output_dir, filename), dpi=200)
        plt.close(f)
    except Exception as error:
        log('Exception occurred while plotting. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def plot_reconstructions(data, samples, filename='reconstructions'):
    try:
        d = 7 if '7x7' in cfg.model else 3
        f, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 7))
        for x, x_rec, ax in zip(data.reshape((-1, d, d)), samples.reshape((-1, d, d)), axs.flat):
            # sns.heatmap(np.concatenate((tonp(x), tonp(x_rec)), 1), ax=ax, cmap='Greens')
            ax.axis('off')
        f.savefig(os.path.join(cfg.output_dir, filename), dpi=200)
        plt.close(f)
    except Exception as error:
        log('Exception occurred while plotting. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def get_glove_embedding(data):
    vectors = torch.load('./glove_embeddings.pkl')
    return torch.FloatTensor(vectors[str(data)])


def compute_offset(task):
    offset1 = task * cfg.continual.n_class_per_task
    offset2 = (task + 1) * cfg.continual.n_class_per_task
    return int(offset1), int(offset2)


def compute_forgetting(accuracies):
    acc_matrix = []
    num_tasks = len(accuracies)
    for index, acc in enumerate(accuracies):
        acc_matrix.append(np.pad(acc, [(0, num_tasks - (index + 1))], mode='constant', constant_values=101))

    acc_matrix = np.array(acc_matrix)
    forgetness = 0
    for t in range(num_tasks-1):
        forgetness += np.max(acc_matrix[t:num_tasks-1,t] - acc_matrix[-1,t])
    avg_forgetness = forgetness / float(num_tasks-1)
    return avg_forgetness
