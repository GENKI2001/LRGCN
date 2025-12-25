from typing import Optional, Tuple
from torch_geometric.typing import Adj, Size, OptTensor, PairTensor

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
import torch.nn as nn
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import reset, glorot, zeros
from torch.nn import ModuleList, Linear, BatchNorm1d

"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/gcn_conv.py
"""
@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[col] * edge_weight * deg_inv_sqrt[col]
        

class GCNConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = True,
                 normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        #glorot(self.weight)
        glorot(self.lin.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, is_add_self_loops: bool = True) -> Tensor:
        original_size = edge_index.shape[1]

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        
        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out, edge_index

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(torch.nn.Module):
    def __init__(self, n_layer, nfeat, nhid, nclass, \
                    normalize=True, is_add_self_loops=True):
        super(GCN, self).__init__()

        self.nclass = nclass
        self.n_layer = n_layer
        self.is_add_self_loops = is_add_self_loops

        self.conv1 = [GCNConv(nfeat, nhid, cached=False, normalize=normalize)]
        self.conv1 += [GCNConv(nhid, nhid, cached=False, normalize=normalize) for _ in range(n_layer-2)]
        self.conv1 += [GCNConv(nhid, nclass, cached=False, normalize=normalize)]
        
        self.conv1 = torch.nn.ModuleList(self.conv1)
        self.reg_params = list(self.conv1.parameters())


    def forward(self, x, edge_index, edge_weight=None, embed=False):
        for i in range(self.n_layer-1):
            x, edge_index = self.conv1[i](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=0.5)
        
        if embed: # when n_layer == 1
            x, edge_index = self.conv1[0](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            return F.relu(x)
        else:
            x, _ = self.conv1[-1](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            return x



class Goodie(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, leaky_alpha=0.01, scaled=False, lamb=1.0, dropout=0.5, num_layers=1):
        super().__init__()
        self.classifier1 = GCN(1, in_channels, hidden_channels, out_channels, normalize=True, is_add_self_loops=False)
        self.classifier2 = GCN(1, out_channels, hidden_channels, out_channels, normalize=True, is_add_self_loops=False)
        self.classifier3 = GCN(1, hidden_channels, hidden_channels, out_channels, normalize=True, is_add_self_loops=False)
        self.leakyrelu = nn.LeakyReLU(leaky_alpha)
        self.attention = nn.Parameter(torch.empty(size=(hidden_channels, 1)))
        glorot(self.attention)
        self.lamb = lamb
        self.scaled = scaled
        self.out_channels = out_channels

    def forward(self, x, lp_embed, edge_index, edge_weight, labels, pseudo_labels, idx_train, weight_mask=None):
        _fp_embed = self.classifier1(x, edge_index, edge_weight, embed=True)
        _lp_embed = self.classifier2(lp_embed, edge_index, edge_weight, embed=True)

        fp_ = self.leakyrelu(torch.mm(_fp_embed, self.attention))
        lp_ = self.leakyrelu(torch.mm(_lp_embed, self.attention))
        values = torch.softmax(torch.cat((fp_, lp_), dim=1), dim=1)
        output_ = values[:, 0:1] * _fp_embed + values[:, 1:2] * _lp_embed
        output = self.classifier3(output_, edge_index, edge_weight)

        pseudocon_loss = 0.0
        if self.lamb != 0.0:
            if self.scaled:
                centroids = []
                C = self.out_channels
                conf = weight_mask if weight_mask is not None else torch.ones(output_.size(0), device=output_.device)
                for c in range(C):
                    idx = (pseudo_labels == c)
                    if idx.sum() == 0:
                        continue
                    z_c = output_[idx]
                    w_c = conf[idx]
                    z_c = (z_c * w_c.view(-1, 1)).sum(0) / (w_c.sum() + 1e-12)
                    centroids.append(z_c)
                if len(centroids) > 1:
                    centroids = torch.stack(centroids, dim=0)
                    pseudocon_loss = self.pseducon_loss(
                        centroids,
                        labels=None,
                        weight_mask=None,
                        scaled=True
                    )
            else:
                pseudocon_loss = self.pseducon_loss(
                    output_,
                    labels=pseudo_labels,
                    weight_mask=weight_mask,
                    scaled=False
                )

        loss_nodeclassification = F.cross_entropy(output[idx_train], labels[idx_train])

        return loss_nodeclassification, pseudocon_loss


    def pseducon_loss(self, features, labels=None, mask=None, temp=0.07, base_temp=0.07, weight_mask=None, scaled=False):
        """
        features:
          scaled=False: [N, D] ノード埋め込み
          scaled=True : [C, D] クラスプロトタイプ
        labels:
          scaled=False: [N] pseudo_labels
          scaled=True : 使わない
        weight_mask:
          scaled=False: [N] 各ノードの信頼度 (train=1.0, pseudo∈(0,1])
          scaled=True : None (未使用)
        """
        features = F.normalize(features, dim=-1)
        device = features.device
        batch_size = features.size(0)

        if scaled:
            logits = torch.matmul(features, features.t()) / temp
            logits_max, _ = logits.max(dim=1, keepdim=True)
            logits = logits - logits_max.detach()

            eye = torch.eye(batch_size, device=device)
            exp_logits = torch.exp(logits) * (1.0 - eye)
            log_prob_pos = torch.diag(logits) - torch.log(exp_logits.sum(1) + 1e-12)

            loss = - (temp / base_temp) * log_prob_pos.mean()
            return loss

        assert labels is not None, "labels (pseudo_labels) is required when scaled=False"

        labels = labels.contiguous().view(-1, 1)
        same_cls = (labels == labels.t()).float().to(device)

        anchor_dot_contrast = torch.matmul(features, features.t()) / temp
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = 1.0 - torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        if weight_mask is None:
            conf = torch.ones(batch_size, device=device)
        else:
            conf = weight_mask.to(device)

        w = (conf.view(-1, 1) * conf.view(1, -1)) * same_cls * logits_mask

        pos_weight_sum = w.sum(1) + 1e-12
        mean_log_prob_pos = (w * log_prob).sum(1) / pos_weight_sum

        loss = - (temp / base_temp) * mean_log_prob_pos.mean()
        return loss