import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from einops import rearrange, repeat, einsum
from torch_sparse import SparseTensor

from mamba_ssm import Mamba as MambaOfficial

class Mamba(nn.Module):
    """
    این Wrapper دقیقاً مثل Mamba قدیمی (v1) رفتار می‌کنه:
    - headdim قبول می‌کنه
    - d_state=1 قبول می‌کنه
    - expand=1 قبول می‌کنه
    - اما از داخل از mamba-ssm 2.2.2 (سریع و بهینه) استفاده می‌کنه
    """
    def __init__(self, d_model, d_state=16, headdim=None, d_conv=4, expand=2, **kwargs):
        super().__init__()
        
        effective_d_state = max(d_state, 8)
        
        self.mamba = MambaOfficial(
            d_model=d_model,
            d_state=effective_d_state,
            d_conv=d_conv,
            expand=expand,
        )
    
    def forward(self, x):
        return self.mamba(x)

class GCN_mamba_liner(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=False):
        super(GCN_mamba_liner, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCN_mamba_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.dropout = args.mamba_dropout
        self.args = args
        self.a = self.args.alpha
        self.b = self.args.graph_weight
        self.lin1 = GCN_mamba_liner(dataset.num_features, args.d_model, with_bias=args.bias)
        self.LayerNorm_1 = torch.nn.LayerNorm(self.args.d_model, eps=1e-12)
        
        
        self.mamba_global_attention = Mamba(
            d_model = args.d_model,
            d_state = 8,
            headdim = 8,     
            d_conv = 4,
            expand = 1,
        )
        
        self.LayerNorm_2 = torch.nn.LayerNorm(self.args.d_model, eps=1e-12)
        self.bn_1 = torch.nn.BatchNorm1d(args.d_model)
        self.layer_num = args.layer_num
        self.mamba = GCN_mamba_block(args, dataset)
        self.norm_1 = RMSNorm(args.d_model)
        self.bn_2 = torch.nn.BatchNorm1d(args.d_model)
        self.lin2 = GCN_mamba_liner(args.d_model, dataset.num_classes, with_bias=args.bias)
        self.bn_3 = torch.nn.BatchNorm1d(args.d_model)

        # لایه‌های اضافی برای VAE-like regularization (mean و log_var)
        self.fc_mu = GCN_mamba_liner(args.d_model, args.d_model, with_bias=args.bias)  # برای mean
        self.fc_logvar = GCN_mamba_liner(args.d_model, args.d_model, with_bias=args.bias)  # برای log_var

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.mamba.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()

    def forward(self, x, adj):
        x_input = x
        adj_t = adj

        x_input = self.lin1(x_input)

        global_attention_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1])
        global_attention_input = F.dropout(global_attention_input, p=self.dropout, training=self.training)

        global_attention_input_flip = torch.flip(global_attention_input, dims=[1])
        global_attention_output = self.mamba_global_attention(global_attention_input)
        global_attention_output_flip = self.mamba_global_attention(global_attention_input_flip)
        global_attention_output = global_attention_output + torch.flip(global_attention_output_flip, dims=[1])

        global_attention_output = F.dropout(global_attention_output, p=self.dropout, training=self.training)
        global_attention_output = F.relu(global_attention_output)
        global_attention_output = torch.squeeze(global_attention_output) * (1-self.a) + self.a * x_input

        x = self.bn_1(x_input)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        all_layers_output = self.mamba(x, adj_t, self.args.layer_num)
        output = (all_layers_output[:,-1,:] + x_input) * self.b + global_attention_output * (1-self.b)
        output = self.bn_2(output)
        output = F.relu(output)
        all_layers_output = F.relu(all_layers_output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        all_layers_output = F.dropout(all_layers_output, p=self.dropout, training=self.training)
        y = self.lin2(output)

        # محاسبه mean و log_var
        mu = self.fc_mu(output)
        log_var = self.fc_logvar(output)
        log_var = torch.clamp(log_var, -10, 10)  # جدید: clamp برای جلوگیری از exploding

        return (mu, log_var), F.log_softmax(y, dim=-1)  # tuple برای VAE

class GCN_mamba_block(torch.nn.Module):
    # بدون تغییر
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.x_proj = GCN_mamba_liner(args.d_inner, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.dropout = args.mamba_dropout
        self.dt_proj = GCN_mamba_liner(args.dt_rank, args.d_inner, with_bias=args.bias)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = GCN_mamba_liner(args.d_inner, args.d_model, with_bias=args.bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x, adj, layer_num):
        alpha = 0.05
        features = [x]
        xi = x
        for i in range(layer_num - 1):
            xi = adj @ xi
            xi = (1-alpha)*xi + alpha*x
            features.append(xi)
        x = torch.stack(features, dim=0).transpose(0, 1)
        y = self.ssm(x, self.args, adj)
        return self.out_proj(y)

    def ssm(self, x, args, adj):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        x_dbl = F.relu(x_dbl)
        x_dbl = F.dropout(x_dbl, p=self.dropout, training=self.training)
        delta, B, C = x_dbl.split(split_size=[args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D, args, adj)

    def selective_scan(self, u, delta, A, B, C, D, args, adj):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(args.layer_num):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            x = F.dropout(x, p=self.dropout, training=self.training)
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight