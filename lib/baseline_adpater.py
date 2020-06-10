from lib.dataset.mnist_variations import MNIST
from lib.config import cfg
from lib.utils import log, Metrics, compute_accuracy, compute_forgetting
import statistics
import importlib
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def run_experiment(method):

    tasks = range(cfg.continual.n_tasks)
    if cfg.continual.shuffle_task:
        tasks = torch.randperm(cfg.continual.n_tasks).tolist()

    cfg.is_cifar_10 = 'cifar10' == cfg.continual.task
    cfg.is_cifar_100 = 'cifar100' == cfg.continual.task
    cfg.is_mini_imagenet = 'mini_imagenet' in cfg.continual.task

    if cfg.is_cifar_10:
        n_inputs = 3 * 32 * 32
        n_outputs = 10
    elif cfg.is_cifar_100 or cfg.is_mini_imagenet:
        n_inputs = 3 * 32 * 32
        n_outputs = 100
    else:
        n_inputs = 28 * 28
        n_outputs = 10

    model = importlib.import_module('lib.baselines.' + method).Net(n_inputs, n_outputs, cfg.continual.n_tasks)
    model.to(cfg.device)
    log(model)
    log('\nRunning experiments using %s.' % method)

    observed_tasks = []
    final_accuracies = []
    individual_acc = []

    for task in tasks:
        train_data = MNIST('./data', task=task, mode='Train', transform=transforms.ToTensor())

        train_dataloader = DataLoader(train_data, batch_size=cfg.continual.batch_size_train,
                                      shuffle=cfg.continual.shuffle_datapoints)
        log('\nLearning task %d.' % task)
        observed_tasks.append(task)

        model.train()

        for epoch in range(cfg.continual.epochs):
            for index, (x, y) in enumerate(train_dataloader):
                x = x.view(-1, n_inputs).to(cfg.device)
                y = y.to(cfg.device)

                model.observe(x, task, y)

        mean_acc, all_acc = test(model, observed_tasks)
        log('Average accuracy for ' + str(observed_tasks) + ' is ' + str(mean_acc))
        final_accuracies.append(mean_acc)
        individual_acc.append(all_acc)
    torch.save(final_accuracies, os.path.join(cfg.output_dir, 'pickles', method + '_final_accuracy.pkl'))
    torch.save(individual_acc, os.path.join(cfg.output_dir, 'pickles', method + '_all_accuracy.pkl'))
    log('Saved to ' + cfg.output_dir)
    log('Final Accuracy:')
    log(final_accuracies)

    log('Average Accuracy: ' + str(statistics.mean(final_accuracies)))
    log('Forgetting: ' + str(compute_forgetting(individual_acc)))


def test(model, observed_tasks):
    model.eval()
    accuracies = []

    if cfg.is_cifar_100 or cfg.is_mini_imagenet or cfg.is_cifar_10:
        n_inputs = 3 * 32 * 32
    else:
        n_inputs = 28 * 28

    for task in observed_tasks:
        test_data = MNIST('./data', task=task, mode='Test', transform=transforms.ToTensor())

        test_dataloader = DataLoader(test_data, batch_size=cfg.continual.batch_size_test,
                                     shuffle=cfg.continual.shuffle_datapoints)

        accuracy_metric = Metrics()
        for index, (x, y) in enumerate(test_dataloader):
            x = x.view(-1, n_inputs).to(cfg.device)
            y = y.to(cfg.device)

            y_pred = model(x, task)
            accuracy = compute_accuracy(y_pred, y)[0].item()
            accuracy_metric.update(accuracy)

        accuracies.append(accuracy_metric.avg)
        log('Accuracy of task %d is %f' % (task, accuracy_metric.avg))
    return statistics.mean(accuracies), accuracies
