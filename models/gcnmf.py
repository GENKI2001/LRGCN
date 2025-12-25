import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import datetime
import utils
from tqdm import trange

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv

class GCNmf(torch.nn.Module):
    def __init__(
        self,
        x,
        adj,
        in_features,
        hidden_features,
        out_features,
        n_components=5,
        dropout=0.5,
        device=None,
    ):
        super(GCNmf, self).__init__()

        if device is None:
            if isinstance(x, torch.Tensor):
                target_device = x.device
            else:
                target_device = torch.device("cpu")
        else:
            target_device = device if isinstance(device, torch.device) else torch.device(device)

        self.gc1 = GCNmfConv(
            in_features=in_features,
            out_features=hidden_features,
            x=x,
            adj=adj,
            n_components=n_components,
            dropout=dropout,
            device=target_device,
        )
        self.gc2 = GCNConv(hidden_features, out_features)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, edge_index, adj=None, adj2=None, label_features=None, **kwargs):
        features = self.gc1(x, adj, adj2)
        logits = self.gc2(features, edge_index)
        return logits

def ex_relu(mu, sigma):
    is_zero = (sigma == 0)
    sigma[is_zero] = 1e-10
    sqrt_sigma = torch.sqrt(sigma)
    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (torch.div(torch.exp(torch.div(- w * w, 2)), np.sqrt(2 * np.pi)) +
                              torch.div(w, 2) * (1 + torch.erf(torch.div(w, np.sqrt(2)))))
    nr_values = torch.where(is_zero, F.relu(mu), nr_values)
    return nr_values


def init_gmm(features, n_components):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    init_x = imp.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag').fit(init_x)
    return gmm


class GCNmfConv(nn.Module):
    def __init__(self, in_features, out_features, x, adj, n_components, dropout, device, bias=True):
        super(GCNmfConv, self).__init__()

        target_device = device if isinstance(device, torch.device) else torch.device(device)
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.dropout = dropout
        self.features = x.detach().cpu().numpy()
        self.logp = Parameter(torch.empty(n_components, device=target_device))
        self.means = Parameter(torch.empty(n_components, in_features, device=target_device))
        self.logvars = Parameter(torch.empty(n_components, in_features, device=target_device))
        self.weight = Parameter(torch.empty(in_features, out_features, device=target_device))
        if not isinstance(adj, torch.Tensor):
            raise TypeError("adj must be a torch.Tensor")
        adj_on_device = adj.to(target_device)
        adj2_on_device = torch.mul(adj_on_device, adj_on_device)
        self.register_buffer("initial_adj", adj_on_device)
        self.register_buffer("initial_adj2", adj2_on_device)
        self.x = x
        self.gmm = None
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=target_device))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        self.gmm = init_gmm(self.features, self.n_components)
        device = self.logp.device
        self.logp.data = torch.as_tensor(np.log(self.gmm.weights_), dtype=torch.float32, device=device)
        self.means.data = torch.as_tensor(self.gmm.means_, dtype=torch.float32, device=device)
        self.logvars.data = torch.as_tensor(np.log(self.gmm.covariances_), dtype=torch.float32, device=device)

    def calc_responsibility(self, mean_mat, variances):
        dim = self.in_features
        log_n = (- 1 / 2) *\
            torch.sum(torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1), 2)\
            - (dim / 2) * np.log(2 * np.pi) - (1 / 2) * torch.sum(self.logvars)
        log_prob = self.logp.unsqueeze(1) + log_n
        return torch.softmax(log_prob, dim=0)

    def forward(self, x, adj=None, adj2=None):
        device = self.weight.device
        x = x.to(device)
        if adj is None:
            adj = self.initial_adj
        elif not isinstance(adj, torch.Tensor):
            raise TypeError("adj must be a torch.Tensor")
        else:
            adj = adj.to(device)

        if adj2 is None:
            adj2 = self.initial_adj2
        elif not isinstance(adj2, torch.Tensor):
            raise TypeError("adj2 must be a torch.Tensor")
        else:
            adj2 = adj2.to(device)

        x_imp = x.repeat(self.n_components, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars)
        mean_mat = torch.where(x_isnan, self.means.repeat((x.size(0), 1, 1)).permute(1, 0, 2), x_imp)
        var_mat = torch.where(x_isnan,
                              variances.repeat((x.size(0), 1, 1)).permute(1, 0, 2),
                              torch.zeros(size=x_imp.size(), device=device, requires_grad=True))

        # dropout
        dropmat = F.dropout(torch.ones_like(mean_mat), self.dropout, training=self.training)
        mean_mat = mean_mat * dropmat
        var_mat = var_mat * dropmat

        transform_x = torch.matmul(mean_mat, self.weight)
        if self.bias is not None:
            transform_x = torch.add(transform_x, self.bias)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []
        for component_x in transform_x:
            conv_x.append(torch.spmm(adj, component_x))
        for component_covs in transform_covs:
            conv_covs.append(torch.spmm(adj2, component_covs))
        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self.calc_responsibility(mean_mat, variances)
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        return expected_x