from lib.dataset import kernels as kernel_dataset
from lib.config import cfg
from models.chunked_vae import CHUNKED_VAE
from lib.utils import log, one_hot, compute_offset
from lib.recall import recall, construct_state_dict_from_weights
from lib.baselines.common import Xavier
from lib.recall import evaluate_classification_model
from lib.dataset.mnist_variations import MNIST
import math
import lib
import models

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist


def encode_weights(task, observed_tasks):
    train_vae(task, observed_tasks)


def train_vae(task, observed_tasks):

    # Dataset
    dataset = kernel_dataset.Kernels(cfg.kernels.dataset_path, task=task)
    dataloader = DataLoader(dataset, batch_size=cfg.kernels.batch_size, shuffle=False)
    cfg.input_size = dataset.get_weight_len()

    # Model
    num_chunks = math.ceil(cfg.input_size / cfg.kernels.chunking.chunk_size)
    cfg.kernels.chunking.last_chunk_size = cfg.input_size % cfg.kernels.chunking.chunk_size
    cfg.kernels.chunking.num_chunks = num_chunks
    model = CHUNKED_VAE(num_chunks).to(cfg.device)

    log('Number of parameters in the classification model: %d' % cfg.input_size)
    log('Number of parameters in VAE: %d' % sum(p.numel() for p in model.parameters() if p.requires_grad))
    log(model)

    # Optimiser
    opt = optim.Adadelta(model.parameters(), lr=0.005, rho=0.9, eps=1e-02)

    # Train
    log('\nTraining VAE:')
    model.train()

    num_epochs = cfg.kernels.epochs
    mean_prior, log_var_prior = torch.zeros(1, cfg.kernels.latent_dimension), torch.ones(1, cfg.kernels.latent_dimension)

    c = one_hot(torch.tensor([task]), cfg.continual.n_tasks).to(cfg.device)
    e = 0
    for e in range(num_epochs):
        for index, x in enumerate(dataloader):
            mean_prior, log_var_prior = train_block_chunk(x, model, task, opt, c, index, e, num_epochs, dataloader)

        if cfg.kernels.intermediate_test:
            checkpoint_location = cfg.output_dir + '/encoded_models/' + 'vae_model.th'
            torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict()}, checkpoint_location)
            log('Saved the model to: %s' % checkpoint_location)
            cfg.recall.model_path = checkpoint_location
            recall(observed_tasks)

    prior_location = cfg.output_dir + '/pickles/prior_' + str(task) + '.pkl'
    torch.save((mean_prior, log_var_prior), prior_location)
    log('Saved the mean and log_var of the prior for task %d to: %s' % (task, prior_location))

    if num_epochs != 0:
        log('Regularising.')
        for task_id in observed_tasks:
            prior_location = cfg.output_dir + '/pickles/prior_' + str(task_id) + '.pkl'
            prior_mean, prior_log_var = torch.load(prior_location)
            task_prior = dist.Normal(prior_mean.to(cfg.device), torch.sqrt(torch.exp(prior_log_var.to(cfg.device))))

            best_acc = -1
            best_weight = None
            for i in range(cfg.kernels.n_pseudo_weights):
                z = task_prior.rsample().squeeze_().to(cfg.device)
                if cfg.kernels.chunking.enable:
                    pseudo_x = decode_chunked_model(model, z)
                else:
                    pseudo_x = model.decoder(z)
                acc, _ = evaluate_classification_model(pseudo_x, [task_id], verbose=False)

                if acc > best_acc:
                    best_acc = acc
                    best_weight = pseudo_x

            for i in range(cfg.kernels.n_finetune_src_models):
                weight = best_weight
                vector = []
                for x in weight:
                    vector.append(x)

                train_block_chunk(vector, model, task, opt, c, logging=False)

    checkpoint_location = cfg.output_dir + '/encoded_models/' + 'vae_model.th'
    torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location)
    log('Saved the model to: %s' % checkpoint_location)
    cfg.recall.model_path = checkpoint_location


def train_block_chunk(x, model, task, opt, c, index=0, e=0, num_epochs=0, dataloader=None, logging=True):
    x = torch.tensor(x).to(cfg.device)

    # Padding zeros for the last chunk
    x = torch.cat((x, torch.zeros(cfg.kernels.chunking.chunk_size - cfg.kernels.chunking.last_chunk_size).to(cfg.device)))

    # Split the parameters 'x' into chunks
    chunks = torch.split(x, cfg.kernels.chunking.chunk_size)

    # Train for each chunks
    for i, chunk in enumerate(chunks):
        x_hat, mean, log_var, mean_prior, log_var_prior = model(chunk, c, i)

        reconstruction_loss = F.mse_loss(x_hat, chunk)
        q = dist.Normal(mean, torch.sqrt(torch.exp(log_var)))
        prior = dist.Normal(mean_prior, torch.sqrt(torch.exp(log_var_prior)))
        kl_div = dist.kl_divergence(q, prior).sum()

        loss = kl_div

        if cfg.kernels.use_reconstruction_loss:
            loss += reconstruction_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    clf_loss = 0
    if cfg.kernels.use_classification_loss:
        clf_loss = get_classification_loss(model, prior, task)
        opt.zero_grad()
        clf_loss.backward()
        opt.step()

    if index % 1 == 0 and logging:
        if cfg.kernels.use_classification_loss:
            log('Epoch: %d/%d \t Iteration: %d/%d \t ELBO: %f (%f + %f) \t Clf Loss: %f' % (e, num_epochs, index,
                                                                             len(dataloader), loss.item(),
                                                                             reconstruction_loss.item(),
                                                                             dist.kl_divergence(q, prior).sum().item(), clf_loss))
        else:
            log('Epoch: %d/%d \t Iteration: %d/%d \t Loss: %f (%f + %f)' % (e, num_epochs, index,
                                                                                            len(dataloader),
                                                                                            loss.item(),
                                                                                            reconstruction_loss.item(),
                                                                                            dist.kl_divergence(q,
                                                                                                               prior).sum().item()))
    return mean_prior, log_var_prior


def decode_chunked_model(model, z):
    x = []
    for chunk_id in range(cfg.kernels.chunking.num_chunks):
        x_hat = model.decoder(z, chunk_id)
        x.append(x_hat)

    x[-1] = x[-1][:cfg.kernels.chunking.last_chunk_size]
    x_hat = torch.cat(x)
    return x_hat


def get_classification_loss(model, prior, task, mode='Train'):
    z = prior.rsample().squeeze_().to(cfg.device)
    weight = decode_chunked_model(model, z)

    if cfg.is_cifar_10:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=10).to(cfg.device)
    elif cfg.is_cifar_100 or cfg.is_mini_imagenet:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=100).to(cfg.device)
    else:
        classification_model = getattr(models.classifiers, cfg.model)().to(cfg.device)

    new_state_dict = construct_state_dict_from_weights(classification_model, weight)
    classification_model.load_state_dict(new_state_dict)
    classification_model.train()

    data = MNIST('./data', task=task, mode=mode, transform=transforms.ToTensor())

    dataloader = DataLoader(data, batch_size=cfg.batch_size_test, shuffle=cfg.continual.shuffle_datapoints)

    loss = 0
    offset1, offset2 = compute_offset(task)
    for index, (x, y) in enumerate(dataloader):
        if 'FC' in cfg.model:
            x = x.view(-1, 28 * 28).to(cfg.device)
        else:
            x = x.to(cfg.device)
        y = y.to(cfg.device)

        y_pred = classification_model(x)

        if cfg.is_cifar_100 or cfg.is_mini_imagenet:
            y_pred_temp = y_pred[:, offset1:offset2]
            y_temp = y - offset1
        else:
            y_pred_temp = y_pred
            y_temp = y

        loss += F.cross_entropy(y_pred_temp, y_temp)

    return loss
