"""
Simple training loop for INR models.

Provides a unified interface for training any INR model.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm


class Trainer:
    """
    Trainer for INR models.
    
    Handles the training loop with progress bars and logging.
    
    Example:
        model = SIREN(in_dim=2, out_dim=3)
        trainer = Trainer(model, lr=1e-4)
        trainer.fit(coords, targets, epochs=1000)
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            model: INR model to train
            lr: Learning rate
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def fit(
        self,
        coords: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        val_coords: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        log_every: int = 100,
        callback: Optional[Callable] = None,
    ):
        """
        Train the model.
        
        Args:
            coords: Input coordinates, shape (N, in_dim)
            targets: Target outputs, shape (N, out_dim)
            epochs: Number of training epochs
            batch_size: Batch size (if None, use full batch)
            val_coords: Validation coordinates (optional)
            val_targets: Validation targets (optional)
            log_every: Log progress every N epochs
            callback: Optional callback function called after each epoch
                     Signature: callback(epoch, train_loss, val_loss)
        """
        # Move data to device
        coords = coords.to(self.device)
        targets = targets.to(self.device)
        
        if val_coords is not None:
            val_coords = val_coords.to(self.device)
            val_targets = val_targets.to(self.device)
        
        # Determine batch size
        if batch_size is None:
            batch_size = len(coords)
        
        # Training loop
        pbar = tqdm(range(epochs), desc='Training')
        for epoch in pbar:
            # Mini-batch training
            indices = torch.randperm(len(coords))
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(coords), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_coords = coords[batch_indices]
                batch_targets = targets[batch_indices]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_coords)
                loss = self.loss_fn(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            val_loss = None
            if val_coords is not None:
                with torch.no_grad():
                    val_outputs = self.model(val_coords)
                    val_loss = self.loss_fn(val_outputs, val_targets).item()
            
            # Logging
            if epoch % log_every == 0:
                log_str = f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}'
                if val_loss is not None:
                    log_str += f', Val Loss = {val_loss:.6f}'
                pbar.set_description(log_str)
            
            # Callback
            if callback is not None:
                callback(epoch, avg_train_loss, val_loss)
    
    def evaluate(self, coords: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Evaluate model on given data.
        
        Args:
            coords: Input coordinates
            targets: Target outputs
        
        Returns:
            loss: MSE loss
        """
        coords = coords.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(coords)
            loss = self.loss_fn(outputs, targets).item()
        
        return loss
