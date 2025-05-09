import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

from models.gnn_model import StockGNN, GraphLevelStockGNN, SectorStockGNN


class GNNTrainer:
    """
    Trainer class for GNN models.
    """
    
    def __init__(self, model_type='node', checkpoints_dir='models/checkpoints'):
        """
        Initialize the GNN trainer.
        
        Args:
            model_type: Type of GNN model ('node', 'graph', or 'sector')
            checkpoints_dir: Directory to save model checkpoints
        """
        self.model_type = model_type
        self.checkpoints_dir = checkpoints_dir
        self.model = None
        self.optimizer = None
        self.scaler = None
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    def prepare_labels(self, price_data, prediction_horizon=5, node_level=True):
        """
        Prepare labels for training (future price changes).
        
        Args:
            price_data: DataFrame with price data (MultiIndex with tickers)
            prediction_horizon: Number of days ahead to predict
            node_level: Whether to create node-level labels (True) or a graph-level label (False)
            
        Returns:
            Labels dictionary (ticker -> label) or a single label for the graph
        """
        # Extract unique tickers
        tickers = price_data.columns.levels[0].unique()
        
        # Dictionary to store labels for each ticker
        ticker_labels = {}
        
        for ticker in tickers:
            try:
                # Get close prices for this ticker
                close_prices = price_data[ticker]['Close']
                
                # Calculate future returns
                future_returns = close_prices.pct_change(periods=prediction_horizon).shift(-prediction_horizon)
                
                # Get the most recent return (which is the label for the current data)
                current_label = future_returns.iloc[-prediction_horizon-1]
                
                # Store the label
                ticker_labels[ticker] = current_label
                
            except Exception as e:
                print(f"Error preparing label for {ticker}: {e}")
                continue
        
        if node_level:
            return ticker_labels
        else:
            # For graph-level prediction, use average return across all stocks
            return np.mean(list(ticker_labels.values()))
    
    def prepare_sector_labels(self, price_data, graph_data, prediction_horizon=5):
        """
        Prepare labels for sector-level prediction.
        
        Args:
            price_data: DataFrame with price data (MultiIndex with tickers)
            graph_data: PyTorch Geometric Data object with sector information
            prediction_horizon: Number of days ahead to predict
            
        Returns:
            Dictionary of labels for each sector
        """
        # Extract tickers and their sectors from graph data
        ticker_to_sector = {graph_data.idx_to_ticker[i]: graph_data.sectors[i] 
                           for i in range(graph_data.num_nodes)}
        
        # Dictionary to store ticker -> label mapping
        ticker_labels = {}
        
        # Calculate future returns for each ticker
        for ticker in graph_data.tickers:
            try:
                # Get close prices for this ticker
                close_prices = price_data[ticker]['Close']
                
                # Calculate future returns
                future_returns = close_prices.pct_change(periods=prediction_horizon).shift(-prediction_horizon)
                
                # Get the most recent return (which is the label for the current data)
                current_label = future_returns.iloc[-prediction_horizon-1]
                
                # Store the label
                ticker_labels[ticker] = current_label
                
            except Exception as e:
                print(f"Error preparing label for {ticker}: {e}")
                continue
        
        # Group labels by sector
        sector_labels = {}
        for ticker, label in ticker_labels.items():
            sector = ticker_to_sector.get(ticker)
            if sector is not None:
                if sector not in sector_labels:
                    sector_labels[sector] = []
                sector_labels[sector].append(label)
        
        # Calculate average return for each sector
        sector_avg_labels = {sector: np.mean(labels) for sector, labels in sector_labels.items()}
        
        return sector_avg_labels
    
    def create_model(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2,
                    dropout=0.2, gnn_type='gcn', num_sectors=11):
        """
        Create the GNN model based on the specified type.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN layer ('gcn', 'gat', or 'sage')
            num_sectors: Number of sectors (for sector-based model)
            
        Returns:
            The created model
        """
        if self.model_type == 'node':
            self.model = StockGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type
            )
        elif self.model_type == 'graph':
            self.model = GraphLevelStockGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type
            )
        elif self.model_type == 'sector':
            self.model = SectorStockGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type,
                num_sectors=num_sectors
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, graph_data, labels, epochs=100, lr=0.001, weight_decay=1e-5,
             batch_size=1, validation_split=0.2, early_stopping=10):
        """
        Train the GNN model.
        
        Args:
            graph_data: PyTorch Geometric Data object
            labels: Labels for training (format depends on model_type)
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            batch_size: Batch size (usually 1 for graph data)
            validation_split: Fraction of data to use for validation
            early_stopping: Number of epochs with no improvement before stopping
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Prepare labels based on model type
        if self.model_type == 'node':
            # Node-level labels
            y = torch.tensor([labels.get(ticker, 0.0) for ticker in graph_data.tickers], 
                            dtype=torch.float).view(-1, 1)
            
        elif self.model_type == 'graph':
            # Graph-level label
            y = torch.tensor([labels], dtype=torch.float).view(-1, 1)
            
        elif self.model_type == 'sector':
            # Just store the sector labels for later use
            y = labels
        
        # Set up the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Set up the loss function (MSE for regression)
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Split data for validation if needed
        use_validation = validation_split > 0 and self.model_type == 'node'
        
        if use_validation and self.model_type == 'node':
            # For node-level models, split the nodes into train/val
            num_nodes = graph_data.num_nodes
            indices = np.arange(num_nodes)
            train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[train_idx] = True
            val_mask[val_idx] = True
        
        # Early stopping setup
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_model_state = None
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.model_type in ['node', 'graph']:
                # Node or graph level prediction
                outputs = self.model(graph_data)
                
                if self.model_type == 'node':
                    if use_validation:
                        # Only use training nodes for loss
                        train_loss = criterion(outputs[train_mask], y[train_mask])
                    else:
                        train_loss = criterion(outputs, y)
                else:
                    # Graph level
                    train_loss = criterion(outputs, y)
                
            else:  # sector model
                # Get sector predictions
                sector_preds = self.model(graph_data)
                
                # Calculate loss for each sector and average
                sector_losses = []
                for sector, pred in sector_preds.items():
                    if sector in y:
                        target = torch.tensor([y[sector]], dtype=torch.float).view(-1, 1)
                        sector_losses.append(criterion(pred, target))
                
                if sector_losses:
                    train_loss = torch.mean(torch.stack(sector_losses))
                else:
                    print("Warning: No sectors with valid predictions and labels")
                    train_loss = torch.tensor(0.0, requires_grad=True)
            
            # Backward pass
            train_loss.backward()
            self.optimizer.step()
            
            # Record training loss
            history['train_loss'].append(train_loss.item())
            
            # Validation
            if use_validation and self.model_type == 'node':
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(graph_data)[val_mask]
                    val_loss = criterion(val_outputs, y[val_mask])
                    history['val_loss'].append(val_loss.item())
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    early_stopping_counter += 1
                
                self.model.train()
                
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                
                if early_stopping_counter >= early_stopping:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}")
        
        # Save the trained model
        self._save_model()
        
        return history
    
    def predict(self, graph_data):
        """
        Make predictions using the trained model.
        
        Args:
            graph_data: PyTorch Geometric Data object
            
        Returns:
            Predictions based on model type
        """
        if self.model is None:
            raise ValueError("Model not created or trained. Call create_model() and train() first.")
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type in ['node', 'graph']:
                predictions = self.model(graph_data)
                
                if self.model_type == 'node':
                    # Return node-level predictions as a dictionary (ticker -> prediction)
                    return {ticker: predictions[i].item() for i, ticker in enumerate(graph_data.tickers)}
                else:
                    # Return graph-level prediction as a single value
                    return predictions.item()
            else:
                # Return sector-level predictions as a dictionary (sector -> prediction)
                sector_preds = self.model(graph_data)
                return {sector: pred.item() for sector, pred in sector_preds.items()}
    
    def _save_model(self):
        """Save the trained model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_type}_gnn_{timestamp}.pt"
        filepath = os.path.join(self.checkpoints_dir, filename)
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'model_config': {
                'input_dim': self.model.input_dim if hasattr(self.model, 'input_dim') 
                        else self.model.node_gnn.input_dim,
                'hidden_dim': self.model.hidden_dim if hasattr(self.model, 'hidden_dim')
                        else self.model.node_gnn.hidden_dim,
                'output_dim': self.model.output_dim if hasattr(self.model, 'output_dim')
                        else 1,
                'num_layers': self.model.num_layers if hasattr(self.model, 'num_layers')
                        else self.model.node_gnn.num_layers,
                'dropout': self.model.dropout if hasattr(self.model, 'dropout')
                        else self.model.node_gnn.dropout,
                'gnn_type': self.model.gnn_type if hasattr(self.model, 'gnn_type')
                        else self.model.node_gnn.gnn_type
            }
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, checkpoint_path):
        """
        Load a trained model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            The loaded model
        """
        checkpoint = torch.load(checkpoint_path)
        model_type = checkpoint['model_type']
        config = checkpoint['model_config']
        
        # Create a new model with the same configuration
        self.model_type = model_type
        self.create_model(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            gnn_type=config['gnn_type']
        )
        
        # Load the model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {checkpoint_path}")
        return self.model
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot the training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()