"""
Training utilities for NIDS models.

This module provides training loops, optimization, and checkpointing functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple, Optional
import json
from pathlib import Path
import time


class NIDSTrainer:
    """
    Trainer class for NIDS models with support for binary and multi-class training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        task: str = 'binary'
    ) -> Tuple[float, Dict]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            task: 'binary' or 'multiclass'
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Handle hybrid model vs single-task model
            if hasattr(self.model, 'binary_head'):
                binary_out, multiclass_out = self.model(batch_x)
                outputs = binary_out if task == 'binary' else multiclass_out
            else:
                outputs = self.model(batch_x)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        task: str = 'binary'
    ) -> Tuple[float, Dict]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            task: 'binary' or 'multiclass'
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'binary_head'):
                    binary_out, multiclass_out = self.model(batch_x)
                    outputs = binary_out if task == 'binary' else multiclass_out
                else:
                    outputs = self.model(batch_x)
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Store predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, labels: list, preds: list) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            labels: True labels
            preds: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        task: str = 'binary',
        class_weights: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            task: 'binary' or 'multiclass'
            class_weights: Class weights for imbalanced data
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        # Setup criterion with class weights
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        print(f"Training {task} classifier for {epochs} epochs...")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, criterion, task)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, criterion, task)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if checkpoint_dir:
                    checkpoint_path = Path(checkpoint_dir) / f"best_model_{task}.pth"
                    self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                    print(f"  [SAVED] Best model (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            print()
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    # Normalize weights
    weights = weights / weights.sum() * len(unique)
    
    return torch.FloatTensor(weights)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Training utilities module loaded successfully!")

