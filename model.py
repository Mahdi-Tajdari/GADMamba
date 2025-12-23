# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution # Keep your original GCN for decoders
from mamba_gcn_modules import GCN_mamba_Net_Encoder # <<< IMPORT THE NEW ENCODER

class MambaGCNEncoder(nn.Module):
    """ A wrapper for the novelty model to be used as an encoder. """
    def __init__(self, feat_size, mamba_args,encoder_mode):
        super().__init__()
        self.encoder = GCN_mamba_Net_Encoder(n_features=feat_size, args=mamba_args, mode=encoder_mode)
        
    # <<< MODIFIED >>>
    def forward(self, x, adj, labels=None, epoch=-1):
        return self.encoder(x, adj, labels, epoch)

# --- Your Decoder classes remain UNCHANGED ---
class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj) 
        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        score = x @ x.T
        return torch.sigmoid(score)

# --- The main DOMINANT model is now much cleaner ---
class Dominant(nn.Module):
    def __init__(self, feat_size, dropout, mamba_args,encoder_mode):
        super(Dominant, self).__init__()
        hidden_size = mamba_args.d_model
        
        self.shared_encoder = MambaGCNEncoder(feat_size, mamba_args, encoder_mode)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    # <<< MODIFIED >>>
    def forward(self, x, adj, labels=None, epoch=-1):
        encoded = self.shared_encoder(x, adj, labels, epoch)
        x_hat = self.attr_decoder(encoded, adj)
        struct_reconstructed = self.struct_decoder(encoded)
        return struct_reconstructed, x_hat
