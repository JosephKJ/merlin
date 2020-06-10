from lib.config import cfg

from torch import nn
import torch


class CHUNKED_VAE(nn.Module):
    def __init__(self, num_chunks):
        super(CHUNKED_VAE, self).__init__()

        input_dim = cfg.kernels.chunking.chunk_size
        hidden_dim = cfg.kernels.chunking.hidden_size
        latent_dim = cfg.kernels.latent_dimension

        self.e1 = nn.Linear(input_dim, hidden_dim)
        self.do1 = nn.Dropout(p=0.3)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.d1 = nn.Linear(2*latent_dim, hidden_dim)
        self.do2 = nn.Dropout(p=0.3)
        self.d2 = nn.Linear(hidden_dim, input_dim)

        self.mu_prior = nn.Linear(cfg.continual.n_tasks, latent_dim)
        self.log_var_prior = nn.Linear(cfg.continual.n_tasks, latent_dim)

        self.chunk_embeddings = nn.Parameter(data=torch.Tensor(num_chunks, latent_dim),requires_grad=True)

    def encoder(self, x):
        act = torch.relu(self.do1(self.e1(x)))
        mean = self.mu(act)
        log_variance = self.log_var(act)
        return mean, log_variance

    def reparameterize(self, mean, log_var):
        sd = torch.exp(0.5 * log_var)
        eps = torch.rand_like(sd)
        return mean + eps * sd

    def decoder(self, z, chunk_id):
        chunk_embed = self.chunk_embeddings[chunk_id]
        comp = torch.cat((chunk_embed, z))
        act = torch.tanh(self.do2(self.d1(comp)))
        weights = torch.tanh(self.d2(act))
        return weights

    def forward(self, x, c, chunk_id):
        m, l_v = self.encoder(x)
        z = self.reparameterize(m, l_v)
        x_hat = self.decoder(z, chunk_id)
        m_prior = self.mu_prior(c)
        l_v_prior = self.log_var_prior(c)

        return x_hat, m, l_v, m_prior, l_v_prior
