from lib.utils import *
from lib.dataset.mnist_variations import MNIST
import models.classifiers
import lib.baselines.common
from lib.recall import construct_state_dict_from_weights
import statistics
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


def train_a_task(task, model_id):
    """
    Define the model, load the dataset, train the model on it.
    :return:
    """
    # Model
    if cfg.is_cifar_10:
        model = getattr(lib.baselines.common, cfg.model)(nclasses=10).to(cfg.device)
    elif cfg.is_cifar_100 or cfg.is_mini_imagenet:
        model = getattr(lib.baselines.common, cfg.model)(nclasses=100).to(cfg.device)
    else:
        model = getattr(models.classifiers, cfg.model)().to(cfg.device)

    # log(model)
    save_location = cfg.output_dir + '/pickles/' + cfg.continual.task + '_initial_model_weight_task_' + str(task) + '.pkl'
    if not os.path.exists(save_location):
        torch.save(model.state_dict(), save_location)
    else:
        model.load_state_dict(torch.load(save_location))

    # Data
    train_data = MNIST('./data', task=task, mode='Train', transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size_train, shuffle=True)

    # Optimizer
    if cfg.continual.task=='rotated_mnist':
        opt = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Train
    model.train()

    tr_acc = []
    test_acc = []
    tr_loss = []
    e = 0
    for e in range(cfg.epochs):
        accuracy_metric = Metrics()
        loss_metric = Metrics()
        for index, (x, y) in enumerate(train_dataloader):
            if 'FC' in cfg.model:
                x = x.view(-1, 28*28).to(cfg.device)
            else:
                x = x.to(cfg.device)

            y = y.to(cfg.device)

            for _ in range(cfg.within_batch_update_count):
                y_pred = model(x)

                loss = F.cross_entropy(y_pred, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

            # Book-keeping
            accuracy = compute_accuracy(y_pred, y)[0].item()
            accuracy_metric.update(accuracy)
            loss_metric.update(loss.item())

            # Training models with random samples from the whole dataset. ('shuffle=True' in DataLoader)
            if index > 0.9 * len(train_dataloader):
                break

        avg_test_accuracy = test(model, [task])
        log('Epochs: %d \t Loss: %f \t Training accuracy: %f \t Test accuracy: %f' %
            (e, loss_metric.avg, accuracy_metric.avg, avg_test_accuracy))

        # Book-keeping after each epoch
        tr_acc.append(accuracy_metric.avg)
        tr_loss.append(loss_metric.avg)
        test_acc.append(avg_test_accuracy)

    pickle_location = cfg.output_dir + '/pickles/' + str(cfg.seed) + '_'
    torch.save(tr_acc, pickle_location+'training_acc.pkl')
    torch.save(tr_loss, pickle_location+'training_loss.pkl')
    torch.save(test_acc, pickle_location+'testing_acc.pkl')

    checkpoint_location = cfg.output_dir + '/models/task_' + str(task) + '_model_' + str(model_id) + '.th'
    torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location)
    cfg.kernels.dataset_path = cfg.output_dir + '/models/'
    log('Saved the model to: %s' % checkpoint_location)


def test(model, tasks, verbose=False, mode='Test'):
    accuracies = []
    for task in tasks:
        test_data = MNIST('./data', task=task, mode=mode, transform=transforms.ToTensor())
        test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size_test, shuffle=cfg.continual.shuffle_datapoints)

        accuracy, _ = evaluate_accuracy(model, test_dataloader, task=task)
        accuracies.append(accuracy)
        if verbose:
            log('Accuracy of task %d is %f' % (task, accuracy))

    acc = statistics.mean(accuracies)
    if verbose:
        log('Average accuracy for ' + str(tasks) + ' is ' + str(acc))

    return acc


def evaluate_accuracy(clf_model, test_dataloader, task=0, compute_loss=False):
    if compute_loss:
        clf_model.train()
    else:
        clf_model.eval()
    accuracy_metric = Metrics()
    loss_metric = Metrics()
    offset1, offset2 = compute_offset(task)
    for index, (x, y) in enumerate(test_dataloader):
        if 'FC' in cfg.model:
            x = x.view(-1, 28 * 28).to(cfg.device)
        else:
            x = x.to(cfg.device)
        y = y.to(cfg.device)

        y_pred = clf_model(x)

        if compute_loss:
            if cfg.is_cifar_100 or cfg.is_mini_imagenet:
                y_pred_temp = y_pred[:, offset1:offset2]
                y_temp = y - offset1
            else:
                y_pred_temp = y_pred
                y_temp = y

            loss = F.cross_entropy(y_pred_temp, y_temp)
            loss_metric.update(loss.item())

        if cfg.is_cifar_100 or cfg.is_mini_imagenet:
            y_pred[:, :offset1].data.fill_(-10e10)
            y_pred[:, offset2:100].data.fill_(-10e10)

        accuracy = compute_accuracy(y_pred, y)[0].item()
        accuracy_metric.update(accuracy)
    return accuracy_metric.avg, loss_metric.avg


def finetune(weight, task, tasks_so_far=None, verbose=True):
    """
    :param weight: classification weights
    :param task: can be -1, if so, tasks_so_far should contains all the task_ids that are seen till now.
    :param tasks_so_far: mutually-exclusive with tasks.
    :param verbose: T/F
    :return: Finetuned weight vector.
    """
    if cfg.is_cifar_10:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=10).to(cfg.device)
    elif cfg.is_cifar_100 or cfg.is_mini_imagenet:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=100).to(cfg.device)
    else:
        classification_model = getattr(models.classifiers, cfg.model)().to(cfg.device)

    if cfg.verbose:
        log(classification_model)

    new_state_dict = construct_state_dict_from_weights(classification_model, weight)
    classification_model.load_state_dict(new_state_dict)

    # Data
    validation_data = MNIST('./data', task=task, mode='Val', transform=transforms.ToTensor(), tasks_so_far=tasks_so_far)

    validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size_train, shuffle=True)

    # Optimizer
    opt = torch.optim.Adam(classification_model.parameters(), lr=cfg.continual.finetune_learning_rate, weight_decay=cfg.weight_decay)

    # Train
    classification_model.train()

    tr_acc = []
    test_acc = []
    tr_loss = []
    for e in range(cfg.continual.n_finetune_epochs):
        accuracy_metric = Metrics()
        loss_metric = Metrics()
        for index, (x, y) in enumerate(validation_dataloader):
            if 'FC' in cfg.model:
                x = x.view(-1, 28 * 28).to(cfg.device)
            else:
                x = x.to(cfg.device)
            y = y.to(cfg.device)

            y_pred = classification_model(x)

            loss = F.cross_entropy(y_pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Book-keeping
            accuracy = compute_accuracy(y_pred, y)[0].item()
            accuracy_metric.update(accuracy)
            loss_metric.update(loss.item())

        avg_test_accuracy = test(classification_model, [task])
        if verbose:
            log('[Fine-tuning] Epochs: %d \t Loss: %f \t Training accuracy: %f \t Test accuracy: %f' %
                (e, loss_metric.avg, accuracy_metric.avg, avg_test_accuracy))

        # Book-keeping after each epoch
        avg_test_accuracy = test(classification_model, [task])
        tr_acc.append(accuracy_metric.avg)
        tr_loss.append(loss_metric.avg)
        test_acc.append(avg_test_accuracy)

    vector = kernels_to_vector(classification_model.state_dict())

    return vector


def kernels_to_vector(model_weights):
    vector = []
    for layer, weight in model_weights.items():
        size = int(torch.prod(torch.tensor(weight.size())).item())
        vector.extend(weight.view(size))
    return vector
