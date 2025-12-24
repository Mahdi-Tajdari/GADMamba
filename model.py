# model.py (updated with neighborhood-focused decoders)

import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution  # Keep your original GCN for decoders
from mamba_gcn_modules import GCN_mamba_Net_Encoder  # <<< IMPORT THE NEW ENCODER

class MambaGCNEncoder(nn.Module):
    """ A wrapper for the novelty model to be used as an encoder. """
    def __init__(self, feat_size, mamba_args, encoder_mode):
        super().__init__()
        self.encoder = GCN_mamba_Net_Encoder(n_features=feat_size, args=mamba_args, mode=encoder_mode)
        
    def forward(self, x, adj, labels=None, epoch=-1):
        return self.encoder(x, adj, labels, epoch)

# --- NEW: Neighborhood-Focused Attribute Decoder ---
class Enhanced_Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, num_neighbors=10):  # Added num_neighbors for sampling
        super(Enhanced_Attribute_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nfeat)
        self.mlp = nn.Linear(nhid, nhid)
        self.dropout = dropout
        self.norm = nn.LayerNorm(nhid)
        self.num_neighbors = num_neighbors  # For neighborhood sampling

    def forward(self, x, adj, attrs):  # Now takes original attrs for neighborhood recon
        x = F.relu(self.mlp(x))
        x = self.norm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Multi-layer GCN with residuals
        residual = x
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x + residual
        
        residual = x
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x + residual
        
        x_hat = self.gc3(x, adj)  # Global reconstruction
        
        # Neighborhood reconstruction: Sample neighbors and reconstruct their attrs
        # Simple random neighbor selection (for efficiency; can use random walk)
        degrees = adj.sum(1)
        neighbor_mask = (degrees > 0).float().unsqueeze(1)
        sampled_neighbors = torch.multinomial(adj, self.num_neighbors, replacement=True)  # Sample neighbors per node
        neigh_attrs = attrs[sampled_neighbors]  # Original neighbor attrs
        neigh_encoded = x[sampled_neighbors]  # Encoded neighbor features
        neigh_recon = F.linear(neigh_encoded, self.gc3.weight.t())  # Simple recon for neighbors (reuse weights)
        
        return x_hat, neigh_recon, neigh_attrs  # Return global + neighborhood for loss

# --- NEW: Spectral-Enhanced Structure Decoder ---
class Enhanced_Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Enhanced_Structure_Decoder, self).__init__()
        self.proj = nn.Linear(nhid, nhid)
        self.gc = GraphConvolution(nhid, nhid)
        self.attn = nn.MultiheadAttention(nhid, num_heads=4, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.proj(x))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.gc(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Spectral component: Compute Laplacian and add low-frequency recon
        laplacian = torch.diag(adj.sum(1)) - adj  # Simple Laplacian
        eigvals, eigvecs = torch.linalg.eigh(laplacian)  # Spectral decomposition (top-k for efficiency if large)
        spectral_recon = eigvecs[:, :10] @ torch.diag(eigvals[:10]) @ eigvecs[:, :10].T  # Low-freq approx
        
        # Attention-based similarity
        x_attn, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x_attn = x_attn.squeeze(0)
        score = torch.matmul(x_attn, x_attn.T) + spectral_recon  # Combine with spectral
        
        return torch.sigmoid(score)

# --- Updated Dominant model ---
class Dominant(nn.Module):
    def __init__(self, feat_size, dropout, mamba_args, encoder_mode):
        super(Dominant, self).__init__()
        hidden_size = mamba_args.d_model
        
        self.shared_encoder = MambaGCNEncoder(feat_size, mamba_args, encoder_mode)
        self.attr_decoder = Enhanced_Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Enhanced_Structure_Decoder(hidden_size, dropout)
    
    def forward(self, x, adj, labels=None, epoch=-1):
        encoded = self.shared_encoder(x, adj, labels, epoch)
        x_hat, neigh_recon, neigh_attrs = self.attr_decoder(encoded, adj, x)  # Pass x as attrs
        struct_reconstructed = self.struct_decoder(encoded, adj)
        return struct_reconstructed, x_hat, neigh_recon, neigh_attrs  # Extra for loss