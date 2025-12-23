# mamba_gcn_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from einops import repeat, einsum
from mamba_ssm import Mamba
from sklearn.metrics import roc_auc_score # <<< ADD THIS IMPORT

# This file contains the core logic from the classification paper's model.
# We will import this into your main anomaly detection model.

class GCN_mamba_liner(torch.nn.Module):
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

class GCN_mamba_block(torch.nn.Module):
    def __init__(self, args):
        super(GCN_mamba_block, self).__init__()
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

    def forward(self, x, adj):
        alpha = 0.05
        features = [x]
        xi = x
        for i in range(self.args.layer_num - 1):
            xi = adj @ xi
            xi = (1 - alpha) * xi + alpha * x
            features.append(xi)
        x_stacked = torch.stack(features, dim=0).transpose(0, 1)
        y = self.ssm(x_stacked)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        x_dbl = F.relu(x_dbl)
        x_dbl = F.dropout(x_dbl, p=self.dropout, training=self.training)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(self.args.layer_num):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            x = F.dropout(x, p=self.dropout, training=self.training)
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y





class GCN_mamba_Net_Encoder(torch.nn.Module):
    def __init__(self, n_features, args, mode='local'):
        super(GCN_mamba_Net_Encoder, self).__init__()
        self.dropout = args.mamba_dropout
        self.args = args
        self.mode = mode
        
        print(f"\n--- GCN Mamba Encoder Initialized in '{self.mode.upper()}' Mode ---\n")

        # لایه‌های مشترک
        self.lin1 = GCN_mamba_liner(n_features, args.d_model, with_bias=args.bias)
        self.bn_2 = torch.nn.BatchNorm1d(args.d_model)

        # لایه‌های مختص حالت Global
        if self.mode == 'global':
            self.mamba_global_attention = Mamba(
                d_model=args.d_model, d_state=8, d_conv=4, expand=1
            )
        
        # لایه‌های مختص حالت Local
        if self.mode == 'local':
            self.bn_1 = torch.nn.BatchNorm1d(args.d_model)
            self.mamba_local = GCN_mamba_block(args)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        if self.mode == 'local':
            self.mamba_local.reset_parameters()

    def forward(self, x, adj, labels=None, epoch=-1):
        # 1. تبدیل خطی اولیه (مشترک در همه حالت‌ها)
        x_input = self.lin1(x)
        
        # --- انتخاب معماری بر اساس حالت ---
        if self.mode == 'global':
            global_attention_input = x_input.unsqueeze(0)
            global_attention_input_flip = torch.flip(global_attention_input, dims=[1])
            global_attention_output = self.mamba_global_attention(global_attention_input)
            global_attention_output_flip = self.mamba_global_attention(global_attention_input_flip)
            global_attention_output = global_attention_output + torch.flip(global_attention_output_flip, dims=[1])
            
            output = F.relu(global_attention_output.squeeze(0))

        elif self.mode == 'local':
            x_local = self.bn_1(x_input)
            x_local = F.relu(x_local)
            x_local = F.dropout(x_local, p=self.dropout, training=self.training)
            all_layers_output = self.mamba_local(x_local, adj)
            local_output = all_layers_output[:, -1, :]
            output = local_output

        elif self.mode == 'linear':
            output = x_input

        else:
            raise ValueError(f"Unknown encoder mode: {self.mode}")
        
        # --- لاگ‌گیری برای تحلیل ---
        if self.training and epoch != -1 and epoch % 20 == 0:
            with torch.no_grad():
                base_norm = x_input.norm().item()
                final_norm = output.norm().item()
                print(
                    f"  [Encoder Norms] Epoch: {epoch} | Mode: {self.mode.upper()} -> "
                    f"Base Norm (after lin1): {base_norm:.3f}, "
                    f"Final Output Norm: {final_norm:.3f}"
                )
        
        # 4. Dropout و نرمال‌سازی نهایی
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.bn_2(output)
        
        return output

