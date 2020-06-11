from lib.train import train_a_task
from lib.consolidate import encode_weights
from lib.recall import recall
from lib.config import cfg
from lib.utils import log, compute_forgetting

import statistics
import os

import torch


def learn_continually():
    log('\nRunning experiments using MERLIN.')

    cfg.is_cifar_10 = 'cifar10' == cfg.continual.task
    cfg.is_cifar_100 = 'cifar100' == cfg.continual.task
    cfg.is_mini_imagenet = 'mini_imagenet' in cfg.continual.task

    tasks = range(cfg.continual.n_tasks)
    if cfg.continual.shuffle_task:
        tasks = torch.randperm(cfg.continual.n_tasks).tolist()

    observed_tasks = []
    final_accuracies = []
    individual_acc = []
    for task in tasks:
        log('\nLearning task %d.' % task)
        observed_tasks.append(task)

        # Train multiple models for a task.
        for model_id in range(cfg.n_models):
            train_a_task(task, model_id)

        # Incrementally consolidate the model.
        encode_weights(task, observed_tasks)

        # Recall the model and compute accuracy on test set.
        acc, all_accuracies = recall(observed_tasks)
        final_accuracies.append(acc)
        individual_acc.append(all_accuracies)
    torch.save(final_accuracies, os.path.join(cfg.output_dir, 'pickles', 'merlin_final_accuracy.pkl'))
    torch.save(individual_acc, os.path.join(cfg.output_dir, 'pickles', 'merlin_all_accuracy.pkl'))
    log('Final Accuracy:')
    log(final_accuracies)

    log('Average Accuracy: ' + str(statistics.mean(final_accuracies)))
    log('Forgetting: ' + str(compute_forgetting(individual_acc)))
