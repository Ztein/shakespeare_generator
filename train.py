import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ShakespeareNet
from data_loader import download_shakespeare
from tqdm import tqdm
import numpy as np
import os
import time

class TextDataset(Dataset):
    def __init__(self, text, sequence_length=100):
        self.text = text
        self.sequence_length = sequence_length
        
        # Create character to index mapping
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Create training sequences
        self.data = [self.char_to_idx[ch] for ch in text]
        
    def __len__(self):
        return len(self.text) - self.sequence_length
        
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + 1:idx + self.sequence_length + 1]
        return (
            torch.tensor(sequence),
            torch.tensor(target)
        )

def get_device():
    """Helper function to get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def get_optimal_workers():
    """Get the optimal number of workers based on the platform and CPU cores"""
    if os.name == 'nt':
        print("Running on Windows, using single worker")
        return 1
    
    # For non-Windows systems, use 80% of available cores
    cores = os.cpu_count()
    optimal_workers = max(1, int(cores * 0.8))
    print(f"Using {optimal_workers} workers ({optimal_workers/cores:.0%} of {cores} CPU cores)")
    return optimal_workers

def train_model(epochs=10, batch_size=128, sequence_length=100):
    # Get data
    text = download_shakespeare()
    dataset = TextDataset(text, sequence_length)
    
    # Get optimal number of workers for the platform
    num_workers = get_optimal_workers()
    
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers, 
                          pin_memory=True)
    
    # Get the best available device
    device = get_device()
    model = ShakespeareNet(dataset.vocab_size).to(device)
    
    # Training setup with improved defaults for M-series chips
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW often works better
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            output, _ = model(data)
            
            loss = criterion(output.view(-1, dataset.vocab_size), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Save model checkpoint if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss
            }, 'best_model.pt')
        
        # Also save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch}.pt')

if __name__ == "__main__":
    train_model() 