"""
Training utilities for GNN models.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import torch

from scripts.common.data_collector import get_data_summary
from scripts.common.graph_data_processor import StockGraphDataProcessor
from models.gnn.gnn_trainer import GNNTrainer

# Configure logger
logger = logging.getLogger(__name__)

def train_gnn_model(
    model_type: str = 'node',
    gnn_type: str = 'gcn',
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    lookback_days: int = 20,
    prediction_horizon: int = 5,
    validation_split: float = 0.2,
    early_stopping: int = 10,
    visualize: bool = False
) -> Union[bool, Dict[str, Any]]:
    """
    Train a GNN model with the specified parameters.
    
    Args:
        model_type: Type of model ('node', 'graph', or 'sector')
        gnn_type: Type of GNN layer ('gcn', 'gat', or 'sage')
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
        lookback_days: Number of days to use for features
        prediction_horizon: Number of days ahead to predict
        validation_split: Fraction of data for validation
        early_stopping: Number of epochs for early stopping
        visualize: Whether to visualize the graph
        
    Returns:
        True if training succeeds, False otherwise.
        If return_model is True, returns the trained model and predictions.
    """
    start_time = time.time()
    logger.info(f"Starting training: {model_type} model with {gnn_type} architecture")
    logger.info(f"Parameters: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, lr={lr}")
    logger.info(f"Training config: epochs={epochs}, lookback_days={lookback_days}, prediction_horizon={prediction_horizon}")
    
    try:
        # Step 1: Create the processed graph data
        logger.info(f"Creating graph data with {lookback_days}-day features")
        processor = StockGraphDataProcessor()
        graph_data = processor.process(lookback_days=lookback_days, 
                                      visualization=visualize)
        
        if graph_data is None:
            logger.error("Failed to create graph data")
            return False
        
        # Log graph characteristics
        logger.info(f"Graph created with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
        
        # Count nodes per sector
        if hasattr(graph_data, 'sectors'):
            sector_counts = {}
            for sector in graph_data.sectors:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            logger.info("Nodes per sector:")
            for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {sector}: {count} stocks")
        
        # Log feature information
        logger.info(f"Node features: shape={graph_data.x.shape}, " +
                   f"min={graph_data.x.min().item():.4f}, " +
                   f"max={graph_data.x.max().item():.4f}, " +
                   f"mean={graph_data.x.mean().item():.4f}")
        
        # Step 2: Get the raw data for creating labels
        logger.info("Loading raw data for labels")
        data_summary = get_data_summary()
        
        if data_summary is None:
            logger.error("Failed to load data summary")
            return False
        
        price_data = data_summary['price_data']
        
        # Step 3: Create the GNN trainer
        logger.info(f"Creating {model_type} GNN trainer")
        trainer = GNNTrainer(model_type=model_type)
        
        # Step 4: Prepare labels based on model type
        logger.info(f"Preparing labels with {prediction_horizon}-day prediction horizon")
        if model_type == 'node':
            labels = trainer.prepare_labels(price_data, 
                                           prediction_horizon=prediction_horizon,
                                           node_level=True)
            
            # Log label statistics
            label_values = np.array(list(labels.values()))
            logger.info(f"Labels: count={len(labels)}, " +
                       f"min={label_values.min():.4f}, " +
                       f"max={label_values.max():.4f}, " +
                       f"mean={label_values.mean():.4f}, " +
                       f"std={label_values.std():.4f}")
            
            # Log distribution of labels
            positive_labels = sum(1 for v in label_values if v > 0)
            negative_labels = sum(1 for v in label_values if v < 0)
            logger.info(f"Label distribution: positive={positive_labels} ({positive_labels/len(label_values):.1%}), " +
                       f"negative={negative_labels} ({negative_labels/len(label_values):.1%})")
            
        elif model_type == 'graph':
            labels = trainer.prepare_labels(price_data, 
                                           prediction_horizon=prediction_horizon,
                                           node_level=False)
            logger.info(f"Graph-level label (market return): {labels:.4f}")
            
        else:  # sector model
            labels = trainer.prepare_sector_labels(price_data, graph_data,
                                                 prediction_horizon=prediction_horizon)
            
            # Log sector label statistics
            logger.info(f"Sector labels: count={len(labels)}")
            for sector, value in sorted(labels.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {sector}: {value:.4f}")
        
        # Step 5: Create the model
        input_dim = graph_data.x.size(1)  # Feature dimension
        num_sectors = len(set(graph_data.sectors)) if hasattr(graph_data, 'sectors') else 0
        
        logger.info(f"Creating model with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim=1")
        logger.info(f"Model architecture: {num_layers} layers of {gnn_type} with dropout={dropout}")
        
        trainer.create_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # Regression model predicting future returns
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            num_sectors=num_sectors
        )
        
        # Log model parameters
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: total={total_params}, trainable={trainable_params}")
        
        # Step 6: Train the model
        logger.info(f"Training model for {epochs} epochs with lr={lr}, weight_decay={weight_decay}")
        if validation_split > 0:
            logger.info(f"Using validation split of {validation_split:.1%} with early stopping (patience={early_stopping})")
        
        # Create a directory for training logs
        os.makedirs("logs", exist_ok=True)
        
        training_start = time.time()
        history = trainer.train(
            graph_data=graph_data,
            labels=labels,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            validation_split=validation_split,
            early_stopping=early_stopping
        )
        training_time = time.time() - training_start
        
        # Log training results
        logger.info(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        if history and 'train_loss' in history:
            initial_loss = history['train_loss'][0]
            final_loss = history['train_loss'][-1]
            best_loss = min(history['train_loss'])
            
            logger.info(f"Training loss: initial={initial_loss:.6f}, final={final_loss:.6f}, best={best_loss:.6f}")
            logger.info(f"Loss reduction: absolute={initial_loss-final_loss:.6f}, relative={(1-final_loss/initial_loss):.1%}")
            
            if 'val_loss' in history and history['val_loss']:
                initial_val = history['val_loss'][0]
                final_val = history['val_loss'][-1]
                best_val = min(history['val_loss'])
                
                logger.info(f"Validation loss: initial={initial_val:.6f}, final={final_val:.6f}, best={best_val:.6f}")
                
                # Check if training might have overfitted
                if final_val > best_val * 1.1:  # 10% worse than best
                    logger.warning("Potential overfitting detected: final validation loss is " +
                                   f"{(final_val/best_val-1):.1%} worse than best validation loss")
        
        # Step 7: Plot and save training history
        logger.info("Creating training history plot")
        os.makedirs("figures", exist_ok=True)
        plot_path = f"figures/{model_type}_{gnn_type}_training_history.png"
        trainer.plot_training_history(
            history,
            save_path=plot_path
        )
        logger.info(f"Training history plot saved to {plot_path}")
        
        # Step 8: Make predictions
        logger.info("Making predictions with trained model")
        predictions = trainer.predict(graph_data)
        
        # Log prediction statistics
        if model_type == 'node':
            pred_values = np.array(list(predictions.values()))
            logger.info(f"Predictions: count={len(predictions)}, " +
                       f"min={pred_values.min():.4f}, " +
                       f"max={pred_values.max():.4f}, " +
                       f"mean={pred_values.mean():.4f}, " +
                       f"std={pred_values.std():.4f}")
            
            # Log distribution of predictions
            positive_preds = sum(1 for v in pred_values if v > 0)
            negative_preds = sum(1 for v in pred_values if v < 0)
            logger.info(f"Prediction distribution: positive={positive_preds} ({positive_preds/len(pred_values):.1%}), " +
                       f"negative={negative_preds} ({negative_preds/len(pred_values):.1%})")
            
            # Save detailed predictions to CSV
            import pandas as pd
            os.makedirs("predictions", exist_ok=True)
            
            pred_df = pd.DataFrame(list(predictions.items()), columns=["Ticker", "Predicted_Return"])
            csv_path = f"predictions/{model_type}_{gnn_type}_predictions.csv"
            pred_df.to_csv(csv_path, index=False)
            logger.info(f"Detailed predictions saved to {csv_path}")
            
        elif model_type == 'graph':
            logger.info(f"Graph-level prediction (market return): {predictions:.4f}")
            
        else:  # sector model
            logger.info(f"Sector predictions: count={len(predictions)}")
            for sector, value in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {sector}: {value:.4f}")
        
        # Display sample predictions
        _display_sample_predictions(predictions, model_type)
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info("GNN training and evaluation completed successfully")
        
        return {
            "trainer": trainer,
            "graph_data": graph_data,
            "history": history,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.exception(f"Error in GNN training: {str(e)}")
        return False

def _display_sample_predictions(predictions, model_type):
    """Display sample predictions based on model type."""
    if model_type == 'node':
        # Sort tickers by predicted return (descending)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Print top 5 positive and negative predictions
        print("\nTop 5 stocks with highest predicted returns:")
        for ticker, pred in sorted_preds[:5]:
            print(f"{ticker}: {pred:.4f}")
        
        print("\nTop 5 stocks with lowest predicted returns:")
        for ticker, pred in sorted_preds[-5:]:
            print(f"{ticker}: {pred:.4f}")
            
    elif model_type == 'graph':
        print(f"\nPredicted market return: {predictions:.4f}")
        
    else:  # sector model
        # Sort sectors by predicted return (descending)
        sorted_sectors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        print("\nSector predictions (descending):")
        for sector, pred in sorted_sectors:
            print(f"{sector}: {pred:.4f}")

def log_hyperparameters_experiment(
    model_type: str,
    gnn_type: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    history: Dict[str, list],
    predictions: Any,
    experiment_name: str = None
):
    """
    Log experiment results for hyperparameter tuning.
    
    Args:
        model_type: Type of model
        gnn_type: Type of GNN layer
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
        history: Training history (dict with train_loss and optionally val_loss)
        predictions: Model predictions
        experiment_name: Optional name for the experiment
    """
    os.makedirs("logs/experiments", exist_ok=True)
    
    exp_name = experiment_name or f"{model_type}_{gnn_type}_h{hidden_dim}_l{num_layers}_d{dropout}"
    
    # Calculate key metrics
    final_train_loss = history['train_loss'][-1]
    best_train_loss = min(history['train_loss'])
    
    if 'val_loss' in history and history['val_loss']:
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
    else:
        final_val_loss = None
        best_val_loss = None
    
    # For node models, calculate prediction statistics
    if model_type == 'node':
        pred_values = np.array(list(predictions.values()))
        pred_mean = pred_values.mean()
        pred_std = pred_values.std()
        pred_min = pred_values.min()
        pred_max = pred_values.max()
    else:
        pred_mean = pred_std = pred_min = pred_max = None
    
    # Create a record for this experiment
    record = {
        "experiment": exp_name,
        "model_type": model_type,
        "gnn_type": gnn_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "final_train_loss": final_train_loss,
        "best_train_loss": best_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to CSV
    import pandas as pd
    log_file = "logs/experiments/hyperparameter_results.csv"
    
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    
    df.to_csv(log_file, index=False)
    
    logger.info(f"Experiment {exp_name} results logged to {log_file}")

if __name__ == "__main__":
    # This allows running the module directly for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('train_gnn.log'),
            logging.StreamHandler()
        ]
    )
    
    # Example usage
    result = train_gnn_model(
        model_type='node',
        gnn_type='gcn',
        epochs=10,  # Short run for testing
        visualize=True
    )
    
    if not result:
        logger.error("Training failed.")