import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """
    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape
        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

class Encoder(nn.Module):
    def __init__(self, input_shape=(2, 65, 65), embed_dim=128, stride=2):
        super().__init__()
        self.stride = stride
        channels, height, width = input_shape
        for _ in range(2):
            height, width = (height - 1) // self.stride + 1, (width - 1) // self.stride + 1 
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=self.stride, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=1)
        self.fc = nn.Linear(height*width*64, embed_dim)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, repr_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim 
        self.fc_dim = action_dim + repr_dim
        self.fc1 = nn.Linear(self.fc_dim, repr_dim*4)
        self.fc2 = nn.Linear(repr_dim*4, repr_dim)
        self.act_fc = nn.Linear(2, self.action_dim)

    def forward(self, s, a):
        a = self.act_fc(a)
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class JepaModel(nn.Module):
    def __init__(self, repr_dim, action_dim, is_training = False):
        super().__init__()
        self.encoder = Encoder((1, 65, 65))
        self.predictor = Predictor(repr_dim, action_dim)
        self.repr_dim = repr_dim 
        self.is_training = is_training 

    def forward(self, states, actions):
        B, L, D = actions.shape
        predicted_embed = []
        encoded_embed = None

        init_state_encoding = self.encoder(states[:, 0, 0, :, :])
        predicted_embed.append(init_state_encoding)
        if self.is_training:
            encoded_embed = []
            for i in range(L):
                encoded_embed.append(self.encoder(states[:, i+1, 0, :, :]))
            encoded_embed = torch.stack(encoded_embed, dim=1)

        for i in range(L):
            sy_hat = self.predictor(predicted_embed[i], actions[:, i, :])
            predicted_embed.append(sy_hat)
        predicted_embed = torch.stack(predicted_embed, dim=1)
        return predicted_embed, encoded_embed