import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def save_model(model, epoch, save_path="checkpoint", file_name="jepa"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"{file_name}_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

def invariance_loss(x, y):
    return F.mse_loss(x, y)

def variance_loss(x, y):
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    return torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def covariance_loss(x,y,d):
    B = x.size(0)
    assert B > 1
    cov_x = (x.T @ x) / (B - 1)
    cov_y = (y.T @ y) / (B - 1)
    return off_diagonal(cov_x).pow(2).sum().div(d) + off_diagonal(cov_y).pow(2).sum().div(d) 

def vicreg_loss(predictor_out, encoder_out):
    predictor_out = predictor_out[:, 1:]
    B, L, D = predictor_out.shape

    inv_loss = F.mse_loss(predictor_out, encoder_out)
    x = predictor_out.contiguous().view(B * L, predictor_out.shape[-1])
    y = encoder_out.contiguous().view(B * L, encoder_out.shape[-1])
    var_loss = variance_loss(x, y) 
    cov_loss = covariance_loss(x, y, D)
    return inv_loss, var_loss, cov_loss