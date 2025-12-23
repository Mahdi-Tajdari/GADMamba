# run.py

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
from sklearn.metrics import roc_auc_score
import argparse
from model import Dominant
from utils import load_anomaly_detection_dataset
import torch.nn.functional as F

def loss_func(adj, A_hat, attrs, X_hat, alpha):
    diff_attribute = F.binary_cross_entropy_with_logits(X_hat, attrs, reduction='none')
    attribute_reconstruction_errors = torch.sum(diff_attribute, 1)
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
    return cost, structure_cost, attribute_cost

def train_dominant(args):
    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    
    if isinstance(adj, scipy.sparse.spmatrix):
        adj = adj.toarray()
    
    adj_tensor = torch.FloatTensor(adj)
    adj_label_tensor = torch.FloatTensor(adj_label)
    attrs_tensor = torch.FloatTensor(attrs)
    
    # Keep original numpy labels for analysis
    numpy_labels = label 

    mamba_args = argparse.Namespace()
    mamba_args.d_model = args.d_model
    mamba_args.d_inner = args.d_inner
    mamba_args.dt_rank = args.dt_rank
    mamba_args.d_state = args.d_state
    mamba_args.layer_num = args.layer_num
    mamba_args.mamba_dropout = args.mamba_dropout
    mamba_args.bias = args.bias

    model = Dominant(feat_size=attrs_tensor.size(1), 
                     dropout=args.dropout, 
                     mamba_args=mamba_args,
                     encoder_mode=args.encoder_mode)
                     
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = model.to(device)
    adj_tensor = adj_tensor.to(device)
    adj_label_tensor = adj_label_tensor.to(device)
    attrs_tensor = attrs_tensor.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        
        # <<< MODIFIED: Pass labels and epoch for analysis >>>
        A_hat, X_hat = model(attrs_tensor, adj_tensor, labels=numpy_labels, epoch=epoch)
        
        loss, struct_loss, feat_loss = loss_func(adj_label_tensor, A_hat, attrs_tensor, X_hat, args.alpha)
        
        l = torch.mean(loss)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()        
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:04d}, train_loss={l.item():.5f}, "
                  f"struct_loss={struct_loss.item():.5f}, feat_loss={feat_loss.item():.5f}")

            model.eval()
            with torch.no_grad():
                # For eval, we don't pass labels
                A_hat, X_hat = model(attrs_tensor, adj_tensor)
                val_loss, _, _ = loss_func(adj_label_tensor, A_hat, attrs_tensor, X_hat, args.alpha)
                score = val_loss.cpu().numpy()
                try:
                    auc = roc_auc_score(label, score)
                    print(f"Epoch: {epoch:04d}, AUC: {auc:.4f}")
                except ValueError:
                    print(f"Epoch: {epoch:04d}, AUC calculation failed (likely only one class present in labels)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog', help='Dataset name')
    parser.add_argument('--epoch', type=int, default=300, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout for decoders')
    parser.add_argument('--alpha', type=float, default=0.8, help='Balance parameter for losses')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW weight decay')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--d_model', type=int, default=64, help='Mamba model dimension (hidden size)')
    parser.add_argument('--d_inner', type=int, default=64, help='Mamba inner dimension')
    parser.add_argument('--dt_rank', type=int, default=32, help='Mamba dt_rank')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--layer_num', type=int, default=3, help='Propagation layers in local mamba')
    parser.add_argument('--mamba_dropout', type=float, default=0.2, help='Dropout inside Mamba blocks')
    parser.add_argument('--bias', action='store_true', help='Use bias in linear layers')
    parser.add_argument('--encoder_mode', type=str, default='local', 
                        choices=['linear', 'local', 'global'], 
                        help='Type of encoder architecture to use.')
    args = parser.parse_args()
    print(args)
    train_dominant(args)
