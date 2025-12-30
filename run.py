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

def grad_hook(module, grad_input, grad_output):
    """
    A backward hook to inspect gradients of a module.
    """
    print(f"--- [GRAD HOOK] Backward pass on {module.__class__.__name__} ---")
    if grad_output and grad_output[0] is not None:
        go = grad_output[0]
        print(f"  Grad Output -> Mean: {go.mean():.6f}, Std: {go.std():.6f}, Norm: {go.norm():.6f}")
    if grad_input and grad_input[0] is not None:
        gi = grad_input[0]
        print(f"  Grad Input  -> Mean: {gi.mean():.6f}, Std: {gi.std():.6f}, Norm: {gi.norm():.6f}")
    print("--- [END GRAD HOOK] ---")
    
def loss_func(adj, A_hat, attrs, X_hat, alpha):
    """
    Calculates the reconstruction loss for attributes and structure.
    """
    diff_attribute = F.binary_cross_entropy_with_logits(X_hat, attrs, reduction='none')
    attribute_reconstruction_errors = torch.sum(diff_attribute, 1)
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
    
    return cost, structure_cost, attribute_cost

def spectral_loss(H, adj, anomaly_scores):
    """
    Calculates a numerically stable spectral loss.
    """
    H_norm = F.normalize(H, p=2, dim=1)
    
    D = torch.diag(adj.sum(1))
    L = D - adj
    
    laplacian_signal = L @ H_norm
    per_node_frequency = torch.sum(laplacian_signal * H_norm, dim=1)
    
    normalized_scores = F.softmax(anomaly_scores.detach(), dim=0)
    
    weighted_frequency = (per_node_frequency * normalized_scores).sum()
    
    return -weighted_frequency

def train_dominant(args):
    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    
    if isinstance(adj, scipy.sparse.spmatrix):
        adj = adj.toarray()
    
    adj_tensor = torch.FloatTensor(adj)
    adj_label_tensor = torch.FloatTensor(adj_label)
    attrs_tensor = torch.FloatTensor(attrs)
    
    numpy_labels = label 

    mamba_args = argparse.Namespace()
    mamba_args.d_model = args.d_model
    mamba_args.d_state = args.d_state
    mamba_args.bias = args.bias
    mamba_args.d_inner = args.d_inner
    mamba_args.dt_rank = args.dt_rank
    mamba_args.layer_num = args.layer_num
    mamba_args.mamba_dropout = args.mamba_dropout

    model = Dominant(feat_size=attrs_tensor.size(1), 
                     dropout=args.dropout, 
                     mamba_args=mamba_args,
                     encoder_mode=args.encoder_mode,
                     ablation_no_mamba=args.ablation_no_mamba)
                     
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = model.to(device)
    adj_tensor = adj_tensor.to(device)
    adj_label_tensor = adj_label_tensor.to(device)
    attrs_tensor = attrs_tensor.to(device)
    
    if args.debug_grads and args.encoder_mode == 'local' and not args.ablation_no_mamba:
        try:
            mamba_module = model.shared_encoder.encoder.mamba_local.mamba
            mamba_module.register_full_backward_hook(grad_hook)
            print("\n[INFO] FULL backward hook registered on the Mamba module.\n")
        except AttributeError:
            print("\n[WARNING] Could not register backward hook.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        
        encoded_features, _ = model.shared_encoder(attrs_tensor, adj_tensor, labels=numpy_labels, epoch=epoch)
        
        x_hat = model.attr_decoder(encoded_features, adj_tensor)
        struct_reconstructed = model.struct_decoder(encoded_features)
        
        recon_loss_per_node, _, _ = loss_func(adj_label_tensor, struct_reconstructed, attrs_tensor, x_hat, args.alpha)
        
        reconstruction_loss = torch.mean(recon_loss_per_node)
        spec_loss = spectral_loss(encoded_features, adj_tensor, recon_loss_per_node)
        
        total_loss = reconstruction_loss + args.spectral_weight * spec_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()        
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:04d}, total_loss={total_loss.item():.5f}, "
                  f"recon_loss={reconstruction_loss.item():.5f}, spec_loss={spec_loss.item():.5f}")

            model.eval()
            with torch.no_grad():
                final_features, pre_mamba_features = model.shared_encoder(attrs_tensor, adj_tensor)

                x_hat_final = model.attr_decoder(final_features, adj_tensor)
                struct_final = model.struct_decoder(final_features)
                val_loss_final, _, _ = loss_func(adj_label_tensor, struct_final, attrs_tensor, x_hat_final, args.alpha)
                score_final = val_loss_final.cpu().numpy()

                x_hat_ablation = model.attr_decoder(pre_mamba_features, adj_tensor)
                struct_ablation = model.struct_decoder(pre_mamba_features)
                val_loss_ablation, _, _ = loss_func(adj_label_tensor, struct_ablation, attrs_tensor, x_hat_ablation, args.alpha)
                score_ablation = val_loss_ablation.cpu().numpy()

                try:
                    auc_final = roc_auc_score(label, score_final)
                    auc_ablation = roc_auc_score(label, score_ablation)
                    print(f"Epoch: {epoch:04d}, AUC (Final): {auc_final:.4f}, AUC (Ablation/Pre-Mamba): {auc_ablation:.4f}")
                except ValueError:
                    print(f"Epoch: {epoch:04d}, AUC calculation failed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog', help='Dataset name')
    parser.add_argument('--epoch', type=int, default=400, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout for decoders')
    parser.add_argument('--alpha', type=float, default=0.8, help='Balance parameter for reconstruction losses')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW weight decay')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--d_model', type=int, default=64, help='Mamba model dimension')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--spectral_weight', type=float, default=0.05, help='Weight for the spectral guidance loss.')
    parser.add_argument('--bias', action='store_true', help='Use bias in linear layers')
    parser.add_argument('--encoder_mode', type=str, default='local', choices=['linear', 'local', 'global'])
    parser.add_argument('--debug_grads', action='store_true', help='Enable gradient hook for debugging.')
    parser.add_argument('--ablation_no_mamba', action='store_true', help="Run without the Mamba core to test other mechanisms.")
    parser.add_argument('--d_inner', type=int, default=64)
    parser.add_argument('--dt_rank', type=int, default=32)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--mamba_dropout', type=float, default=0.2)

    args = parser.parse_args()
    print(args)
    train_dominant(args)
