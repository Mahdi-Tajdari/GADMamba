# mamba_gcn_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances

# ==========================================================================================
# DEBUGGING UTILITIES
# ==========================================================================================

def log_activations(layer_output, epoch, mode):
    """
    Logs statistics of layer activations and saves a histogram.
    """
    try:
        acts = layer_output.detach().cpu().numpy().flatten()
        stats_text = (f"Epoch {epoch} | Mode: {mode} | Activations Stats -> "
                      f"Mean: {acts.mean():.4f}, Std: {acts.std():.4f}, "
                      f"Min: {acts.min():.4f}, Max: {acts.max():.4f}")
        print(stats_text)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(acts, bins=50, kde=True)
        plt.title(f"Activations Histogram - {mode.upper()} Mamba Epoch {epoch}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.savefig(f"acts_epoch_{epoch}_{mode}.png")
        plt.close()
        print(f"[DEBUG] Activation histogram saved to 'acts_epoch_{epoch}_{mode}.png'")

    except Exception as e:
        print(f"[ERROR] Failed to log activations: {e}")

def compute_mad(embeddings):
    """
    Computes the Mean Average Distance (MAD) using cosine distance.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
        
    dists = cosine_distances(embeddings)
    return dists.mean()

# ==========================================================================================
# MODULES
# ==========================================================================================

class GCN_mamba_liner(torch.nn.Module):
    def __init__(self, in_features, out_features, with_bias=True):
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
        nn.init.kaiming_normal_(self.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SelectiveViewFusionBlock(nn.Module):
    def __init__(self, args, tau=1.0, ablation_no_mamba=False):
        super().__init__()
        dim = args.d_model
        
        self.ablation_no_mamba = ablation_no_mamba
        
        if not self.ablation_no_mamba:
            self.mamba = Mamba(d_model=dim, d_state=args.d_state, d_conv=4, expand=2)
        
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        
        self.proj_v1 = GCN_mamba_liner(dim, dim)
        self.proj_v2 = GCN_mamba_liner(dim, dim)
        
        self.gate_fusion = GCN_mamba_liner(dim * 2, dim)
        self.gate_nspl_logits = GCN_mamba_liner(dim, 2)
        self.tau = tau

    def forward(self, x, adj):
        residual = x
        x_norm = self.norm_in(x)
        
        nspl_logits = self.gate_nspl_logits(x_norm) 
        nspl_gate = F.gumbel_softmax(nspl_logits, tau=self.tau, hard=True)[:, 1]
        selective_adj = adj * nspl_gate.unsqueeze(1) * nspl_gate.unsqueeze(0)
        
        view1 = x_norm
        if adj.is_sparse:
            view2 = torch.sparse.mm(selective_adj, x_norm)
        else:
            view2 = torch.mm(selective_adj, x_norm)
            
        fusion_gate_input = torch.cat([view1, view2], dim=-1)
        fusion_gate_values = torch.sigmoid(self.gate_fusion(fusion_gate_input))
        fused_view = fusion_gate_values * self.proj_v1(view1) + (1 - fusion_gate_values) * self.proj_v2(view2)
        
        if self.ablation_no_mamba:
            mamba_output = fused_view
        else:
            mamba_input = fused_view.unsqueeze(0)
            mamba_output = self.mamba(mamba_input).squeeze(0)
        
        output = self.norm_out(mamba_output + residual)
        return output

class GCN_mamba_Net_Encoder(torch.nn.Module):
    def __init__(self, n_features, args, mode='local', ablation_no_mamba=False):
        super(GCN_mamba_Net_Encoder, self).__init__()
        self.dropout = args.mamba_dropout
        self.args = args
        self.mode = mode
        
        if ablation_no_mamba:
            print("\n--- GCN Mamba Encoder Initialized in ABLATION MODE (NO MAMBA) ---\n")
        else:
            print(f"\n--- GCN Mamba Encoder Initialized in '{self.mode.upper()}' Mode (Full Model) ---\n")

        self.lin1 = GCN_mamba_liner(n_features, args.d_model, with_bias=args.bias)
        self.bn_2 = torch.nn.BatchNorm1d(args.d_model)

        if self.mode == 'local':
            self.mamba_local = SelectiveViewFusionBlock(args, ablation_no_mamba=ablation_no_mamba)
        elif self.mode == 'global':
            self.mamba_global_attention = Mamba(d_model=args.d_model, d_state=8, d_conv=4, expand=1)

    def forward(self, x, adj, labels=None, epoch=-1):
        x_input = self.lin1(x)
        
        if self.mode == 'local':
            mamba_processed = self.mamba_local(x_input, adj)
            output = F.dropout(mamba_processed, p=self.dropout, training=self.training)
            output = self.bn_2(output)
            
            if self.training and epoch != -1 and epoch % 50 == 0:
                print(f"\n--- [DEBUG] Epoch {epoch}, Mode: {self.mode.upper()} ---")
                print("[DEBUG] Logging activation statistics and histogram...")
                log_activations(output.detach(), epoch, self.mode)
                print("[DEBUG] Computing Mean Average Distance (MAD)...")
                mad_input = compute_mad(x_input.detach())
                mad_output = compute_mad(output.detach())
                print(f"[DEBUG] MAD Input: {mad_input:.4f}, MAD Output: {mad_output:.4f}")
                print("--- [END DEBUG] ---\n")
            
            return output, x_input
        
        output = F.dropout(x_input, p=self.dropout, training=self.training)
        output = self.bn_2(output)
        return output, x_input
