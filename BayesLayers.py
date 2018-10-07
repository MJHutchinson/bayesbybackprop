import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(torch.abs(sigma)) - (x - mu)**2 / (2 * sigma**2)



class BayesLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.variational_loss = 0
        self.prior_loss = 0

    def get_losses(self):
        return self.variational_loss, self.prior_loss

    def sample_weights(self):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


class BayesFCLayer(BayesLayer):

    def __init__(self, n_input, n_output, prior_sigma):
        super(BayesFCLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.prior_sigma = prior_sigma
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, self.prior_sigma))
        self.W_rho = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, self.prior_sigma))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, self.prior_sigma))
        self.b_rho = nn.Parameter(torch.Tensor(n_output).normal_(0, self.prior_sigma))

    def forward(self, X, fix_weights=False):
        if fix_weights:
            return torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_output)

        W, b = self.sample_weights()

        self.variational_loss = log_gaussian(W, self.W_mu, torch.log(1 + torch.exp(self.W_rho))).sum() + log_gaussian(b, self.b_mu, torch.log(1 + torch.exp(self.b_rho))).sum()
        self.prior_loss = log_gaussian(W, 0, (torch.ones(W.size())*self.prior_sigma).cuda()).sum() + log_gaussian(b, 0, (torch.ones(b.size())*self.prior_sigma).cuda()).sum()

        return torch.mm(X, W) + b.expand(X.size()[0], self.n_output)

    def sample_weights(self):
        epsilon_w = Variable(torch.Tensor(self.W_mu.size()).normal_(0, self.prior_sigma).cuda())
        epsilon_b = Variable(torch.Tensor(self.b_mu.size()).normal_(0, self.prior_sigma).cuda())

        return (self.W_mu + torch.log(1 + torch.exp(self.W_rho)) * epsilon_w), \
            (self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * epsilon_b)


# class BayesWrapper(nn.Module):
#
#     def __init__(self, model, n_samples):
#         super(BayesWrapper, self).__init__()
#         self.model = model
#         self.n_samples = n_samples
#
#     def forward(self, *input):
#         for _ in self.n_samples:
#
#
#     def get_losses(self):
#         variational_loss, prior_loss = 0, 0
#         for module in self._modules.values():
#             if isinstance(module, BayesLayer):
#                 vl, pl = module.get_losses()
#                 variational_loss += vl
#                 prior_loss += pl
#
#         return variational_loss, prior_loss

class BayesMLP(nn.Module):

    def __init__(self, n_input, n_output, prior_sigma, num_hidden_units):
        super(BayesMLP, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.prior_sigma = prior_sigma
        self.num_hidden_units = num_hidden_units

        self.l1 = BayesFCLayer(n_input, num_hidden_units, prior_sigma)
        self.relu1 = nn.ReLU()
        self.l2 = BayesFCLayer(num_hidden_units, num_hidden_units, prior_sigma)
        self.relu2 = nn.ReLU()
        self.l3 = BayesFCLayer(num_hidden_units, n_output, prior_sigma)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, fix_weights=False):
        X = X.view(X.size()[0], -1)
        output = self.relu1(self.l1(X, fix_weights))
        output = self.relu2(self.l2(output, fix_weights))
        output = self.softmax(self.l3(output, fix_weights))
        return output

    def get_losses(self):
        variational_loss, prior_loss = 0, 0
        for module in self._modules.values():
            if isinstance(module, BayesLayer):
                vl, pl = module.get_losses()
                variational_loss += vl
                prior_loss += pl
        return variational_loss, prior_loss

    def bayes_sample_loss(self, x, y, n_samples):
        log_qw, log_pw, log_likelihood = 0., 0., 0.

        rows = torch.LongTensor(np.arange(0, y.size()[0]))
        i = torch.stack([rows, y], 0)
        y_ohe = torch.sparse.FloatTensor(i, torch.ones(y.size()[0])).to_dense().cuda()

        for _ in range(n_samples):
            y_hat = self(x.cuda())

            sample_log_qw, sample_log_pw = self.get_losses()
            sample_log_likelihood = log_gaussian(y_ohe, y_hat, (torch.ones(y_hat.size()) * self.prior_sigma).cuda()).sum()  # -F.nll_loss(torch.log(y_hat), y.cuda(), reduction='sum')

            log_qw += sample_log_qw / n_samples
            log_pw += sample_log_pw / n_samples
            log_likelihood += sample_log_likelihood / n_samples

        return log_qw, log_pw, log_likelihood