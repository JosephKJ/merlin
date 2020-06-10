import models.classifiers
from lib.config import cfg
from lib.utils import log, Metrics, compute_accuracy, compute_offset
import lib
from models.chunked_vae import CHUNKED_VAE

import statistics
import numpy as np

import torch
from torch.utils.data import DataLoader
from lib.dataset.mnist_variations import MNIST
from torchvision import transforms
import torch.distributions as dist


def recall(observed_tasks):
    if cfg.recall.cumulative_prior:
        weights = get_weights_from_chunked_vae_cumulative_prior(observed_tasks)
        acc, all_accuracies = ensemble_and_evaluate_cumulative_prior(weights, observed_tasks)
    else:
        weights = get_weights_from_chunked_vae(observed_tasks)
        acc, all_accuracies = ensemble_and_evaluate(weights, observed_tasks)

    log('Test Accuracy: %f' % acc)
    return acc, all_accuracies


def ensemble_and_evaluate(weights, observed_tasks):
    log('Analysing weights for ensembling.')

    classification_models = []
    for task, task_weight in enumerate(weights):
        classification_models_ensemble = []
        for weight in task_weight:
            weight_ft = torch.tensor(lib.train.finetune(weight, task, verbose=False))
            acc, model = evaluate_classification_model(weight_ft, [task], verbose=False, mode='Test')
            if acc > cfg.kernels.ensembling.min_clf_accuracy:
                log('[Task: %d] Individual accuracies: %f' % (task, acc))
                classification_models_ensemble.append(model)
        classification_models.append(classification_models_ensemble)

    log('Ensembling results from %d tasks.' % len(classification_models))

    accuracies = []
    for task in observed_tasks:
        acc = ensembled_prediction_for_a_task(task, classification_models[task])
        accuracies.append(acc)

    acc = statistics.mean(accuracies)
    log('Average accuracy for ' + str(observed_tasks) + ' is ' + str(acc))
    return acc, accuracies


def ensembled_prediction_for_a_task(task, clf_models):
    test_data = MNIST('./data', task=task, mode='Test', transform=transforms.ToTensor())

    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size_test,
                                 shuffle=cfg.continual.shuffle_datapoints)

    accuracy_metric = Metrics()
    offset1, offset2 = compute_offset(task)

    for idx, (x, y) in enumerate(test_dataloader):
        if 'FC' in cfg.model:
            x = x.view(-1, 28 * 28).to(cfg.device)
        else:
            x = x.to(cfg.device)
        y = y.to(cfg.device)

        if cfg.is_cifar_100 or cfg.is_mini_imagenet:
            y_pred = torch.zeros((x.size()[0], 100)).to(cfg.device)
        else:
            y_pred = torch.zeros((x.size()[0], 10)).to(cfg.device)


        for model in clf_models:
            # Model Aggregation: Max Voting
            output = model(x)
            if cfg.is_cifar_100 or cfg.is_mini_imagenet:
                output[:, :offset1].data.fill_(-10e10)
                output[:, offset2:100].data.fill_(-10e10)
            elif cfg.is_cifar_10:
                output[:, :offset1].data.fill_(-10e10)
                output[:, offset2:10].data.fill_(-10e10)
            y_pred += output

        accuracy = compute_accuracy(y_pred, y)[0].item()
        accuracy_metric.update(accuracy)

    log('Accuracy of task %d is %f' % (task, accuracy_metric.avg))
    return accuracy_metric.avg


def evaluate_classification_model(weight, observed_tasks, verbose=True, mode='Test'):

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
    classification_model.eval()

    accuracy = lib.train.test(classification_model, observed_tasks, verbose=verbose, mode=mode)
    return accuracy, classification_model


def construct_state_dict_from_weights(model, weights):
    new_state_dict = {}
    index = 0

    for layer, weight in model.state_dict().items():
        offset = int(torch.prod(torch.tensor(weight.size())).item())
        new_state_dict[layer] = weights[index:index + offset].view(weight.size())
        index = index + offset

    return new_state_dict


def ensemble_and_evaluate_cumulative_prior(weights, observed_tasks):
    log('Analysing weights for ensembling.')

    classification_models = []
    for index, weight in enumerate(weights):
        weight_ft = torch.tensor(lib.train.finetune(weight, -1, tasks_so_far=observed_tasks, verbose=False))
        acc, model = evaluate_classification_model(weight_ft, observed_tasks, verbose=False, mode='Test')
        if acc > cfg.kernels.ensembling.min_clf_accuracy:
            log('[%d / %d] Individual accuracies: %f' % (index, len(weights), acc))
            classification_models.append(model)

    log('Ensembling the models.')
    accuracies = []
    for task in observed_tasks:
        acc = ensembled_prediction_for_a_task(task, classification_models)
        accuracies.append(acc)

    acc = statistics.mean(accuracies)
    log('Average accuracy for ' + str(observed_tasks) + ' is ' + str(acc))
    return acc, accuracies


def get_weights_from_chunked_vae_cumulative_prior(observed_tasks):

    # Loading model
    model = CHUNKED_VAE(cfg.kernels.chunking.num_chunks).to(cfg.device)
    checkpoint = torch.load(cfg.recall.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = []
    for task in observed_tasks:
        # Loading task-specific prior
        prior_location = cfg.output_dir + '/pickles/prior_' + str(task) + '.pkl'
        prior_mean, _ = torch.load(prior_location)
        mean.append(prior_mean.cpu().data.numpy()[0])

    prior_mean = torch.FloatTensor(np.average(mean, axis=0))
    prior_log_var = torch.ones_like(prior_mean)

    prior = dist.Normal(prior_mean, torch.sqrt(torch.exp(prior_log_var)))
    ensemble_weights = []
    for i in range(cfg.kernels.ensembling.max_num_of_models):
        z = prior.rsample().squeeze_().to(cfg.device)
        weight = lib.consolidate.decode_chunked_model(model, z).to('cpu')
        ensemble_weights.append(weight)

    return ensemble_weights


def get_weights_from_chunked_vae(observed_tasks):
    weights = []

    # Loading model
    model = CHUNKED_VAE(cfg.kernels.chunking.num_chunks).to(cfg.device)
    checkpoint = torch.load(cfg.recall.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for task in observed_tasks:
        # Loading task-specific prior
        prior_location = cfg.output_dir + '/pickles/prior_' + str(task) + '.pkl'
        prior_mean, prior_log_var = torch.load(prior_location)
        prior = dist.Normal(prior_mean, torch.sqrt(torch.exp(prior_log_var)))

        ensemble_weights = []
        for i in range(cfg.kernels.ensembling.max_num_of_models):
            z = prior.rsample().squeeze_().to(cfg.device)
            weight = lib.consolidate.decode_chunked_model(model, z).to('cpu')
            ensemble_weights.append(weight)

        weights.append(ensemble_weights)
    return weights