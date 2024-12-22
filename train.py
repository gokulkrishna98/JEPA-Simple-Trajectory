import torch
import torch.optim as optim

from  tqdm import tqdm
from torch.utils.data import DataLoader

from main import get_device
from dataset import TrajectoryDataset
from models import JepaModel
from utils import invariance_loss, covariance_loss, variance_loss, save_model, vicreg_loss

def train(model, dataloader, device, optimizer, num_epocs):
    model = model.to(device)
    model.train()
    for epoch in tqdm(range(num_epocs), desc='Epoch:'):
        epoch_loss = 0.0

        mse_loss, var_loss, cov_loss = 0.0, 0.0, 0.0
        for states, actions in tqdm(dataloader, desc='Training across the dataset for this epoch'):
            states = states.to(device)
            actions = actions.to(device)
            B, L, _ = actions.shape 
            D = model.repr_dim

            predictor_out, encoded_out = model(states, actions)
            mse_loss_l, var_loss_l, cov_loss_l  = vicreg_loss(predictor_out, encoded_out) 
            loss = 25.0*mse_loss_l + 25.0*var_loss_l + cov_loss_l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            epoch_loss += loss.item()
            mse_loss += mse_loss_l.item()
            var_loss += var_loss_l.item()
            cov_loss += cov_loss_l.item()
        
        save_model(model, epoch)
        print(f"Epoch {epoch+1}, MSE Loss: {mse_loss/ len(dataloader):.10f}")
        print(f"Epoch {epoch+1}, VAR Loss: {var_loss/ len(dataloader):.10f}")
        print(f"Epoch {epoch+1}, COV Loss: {cov_loss/ len(dataloader):.10f}")
        print(f"Epoch {epoch+1}, TOTAL Loss: {epoch_loss / len(dataloader):.10f}") 

if __name__ == "__main__":
    dataset = TrajectoryDataset("./dataset/dataset", "states.npy", "actions.npy")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = JepaModel(128, 16, is_training=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train(model, dataloader, get_device(), optimizer, 10)
