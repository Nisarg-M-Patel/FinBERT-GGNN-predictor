#!/usr/bin/env python
"""
Prediction script for the S&P 500 GNN model.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gnn_prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('gnn_predictor')

# Import local modules
from scripts.common.data_collector import get_data_summary
from models.gnn_trainer import GNNTrainer
from scripts.common.graph_data_processor import StockGraphDataProcessor


def predict_with_gnn(checkpoint_path, lookback_days=20):
    """
    Make predictions using a trained GNN model.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        lookback_days: Number of days to use for features
    """
    try:
        # Step 1: Load the trained model
        logger.info(f"Loading model from {checkpoint_path}")
        trainer = GNNTrainer()
        model = trainer.load_model(checkpoint_path)
        
        # Step 2: Create the graph data
        logger.info(f"Creating graph data with {lookback_days}-day features")
        processor = StockGraphDataProcessor()
        graph_data = processor.process(lookback_days=lookback_days, visualization=False)
        
        if graph_data is None:
            logger.error("Failed to create graph data")
            return
        
        # Step 3: Make predictions
        logger.info("Making predictions")
        predictions = trainer.predict(graph_data)
        
        # Step 4: Process and display predictions based on model type
        if trainer.model_type == 'node':
            # Sort tickers by predicted return (descending)
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Create a dataframe for easier analysis
            pred_df = pd.DataFrame(sorted_preds, columns=['Ticker', 'Predicted_Return'])
            
            # Print top 10 positive and negative predictions
            print("\nTop 10 stocks with highest predicted returns:")
            print(pred_df.head(10).to_string(index=False))
            
            print("\nTop 10 stocks with lowest predicted returns:")
            print(pred_df.tail(10).to_string(index=False))
            
            # Get sector information
            sector_data = pd.read_csv(os.path.join(processor.data_dir, "sp500_sectors.csv"))
            
            # Merge predictions with sector info
            pred_df = pd.merge(pred_df, sector_data, left_on='Ticker', right_on='Symbol', how='left')
            
            # Group by sector and calculate average predicted return
            sector_preds = pred_df.groupby('Sector')['Predicted_Return'].mean().sort_values(ascending=False)
            
            print("\nAverage predicted returns by sector:")
            print(sector_preds.to_string())
            
            # Save predictions to CSV
            output_dir = "predictions"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"stock_predictions_{timestamp}.csv")
            pred_df.to_csv(output_file, index=False)
            
            print(f"\nPredictions saved to {output_file}")
        
        elif trainer.model_type == 'graph':
            print(f"\nOverall market prediction:")
            print(f"Predicted market return: {predictions:.4f}")
            
            # Save prediction to a text file
            output_dir = "predictions"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"market_prediction_{timestamp}.txt")
            
            with open(output_file, 'w') as f:
                f.write(f"Prediction made on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Predicted market return: {predictions:.4f}\n")
            
            print(f"\nPrediction saved to {output_file}")
        
        else:  # sector model
            # Sort sectors by predicted return (descending)
            sorted_sectors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Create a dataframe for easier analysis
            sector_df = pd.DataFrame(sorted_sectors, columns=['Sector', 'Predicted_Return'])
            
            print("\nSector predictions (ranked by predicted return):")
            print(sector_df.to_string(index=False))
            
            # Save predictions to CSV
            output_dir = "predictions"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"sector_predictions_{timestamp}.csv")
            sector_df.to_csv(output_file, index=False)
            
            print(f"\nPredictions saved to {output_file}")
        
        # Create a visualization of the predictions
        logger.info("Creating prediction visualization")
        _visualize_predictions(predictions, trainer.model_type, graph_data)
        
        logger.info("Prediction process completed successfully")
        return predictions
        
    except Exception as e:
        logger.exception(f"Error making predictions: {str(e)}")
        return None


def _visualize_predictions(predictions, model_type, graph_data=None):
    """
    Create visualizations for the predictions.
    
    Args:
        predictions: Prediction results
        model_type: Type of model ('node', 'graph', or 'sector')
        graph_data: PyTorch Geometric Data object (for node-level predictions)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Create predictions directory
    output_dir = os.path.join("figures", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create different visualizations based on model type
    if model_type == 'node':
        # Sort predictions
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        tickers, values = zip(*sorted_preds)
        
        # Histogram of predicted returns
        plt.figure(figsize=(12, 6))
        sns.histplot(values, bins=30, kde=True)
        plt.title("Distribution of Predicted Stock Returns")
        plt.xlabel("Predicted Return")
        plt.ylabel("Frequency")
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Add mean and median lines
        mean_return = np.mean(values)
        median_return = np.median(values)
        plt.axvline(x=mean_return, color='g', linestyle='-', alpha=0.7, 
                    label=f'Mean: {mean_return:.4f}')
        plt.axvline(x=median_return, color='b', linestyle='-', alpha=0.7,
                    label=f'Median: {median_return:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "return_distribution.png"), dpi=300)
        plt.close()
        
        # Top/bottom 20 stocks
        plt.figure(figsize=(14, 8))
        
        # Top 20
        top_tickers = tickers[:20]
        top_values = values[:20]
        
        plt.subplot(2, 1, 1)
        bars = plt.barh(range(len(top_tickers)), top_values, align='center')
        plt.yticks(range(len(top_tickers)), top_tickers)
        plt.gca().invert_yaxis()  # Top ticker at the top
        plt.title("Top 20 Stocks by Predicted Return")
        plt.xlabel("Predicted Return")
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            if top_values[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Bottom 20
        bottom_tickers = tickers[-20:]
        bottom_values = values[-20:]
        
        plt.subplot(2, 1, 2)
        bars = plt.barh(range(len(bottom_tickers)), bottom_values, align='center')
        plt.yticks(range(len(bottom_tickers)), bottom_tickers)
        plt.gca().invert_yaxis()  # Bottom ticker at the top of its subplot
        plt.title("Bottom 20 Stocks by Predicted Return")
        plt.xlabel("Predicted Return")
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            if bottom_values[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_bottom_stocks.png"), dpi=300)
        plt.close()
        
        # Sector analysis
        if graph_data and hasattr(graph_data, 'sectors'):
            # Create a dataframe with predictions and sectors
            sectors = graph_data.sectors
            ticker_to_sector = {graph_data.idx_to_ticker[i]: sectors[i] 
                               for i in range(graph_data.num_nodes)}
            
            sector_pred_data = []
            for ticker, pred in predictions.items():
                sector = ticker_to_sector.get(ticker)
                if sector:
                    sector_pred_data.append((ticker, pred, sector))
            
            sector_df = pd.DataFrame(sector_pred_data, columns=['Ticker', 'Prediction', 'Sector'])
            
            # Group by sector
            sector_stats = sector_df.groupby('Sector')['Prediction'].agg(['mean', 'std', 'count'])
            sector_stats = sector_stats.sort_values('mean', ascending=False)
            
            # Plot average returns by sector
            plt.figure(figsize=(12, 8))
            bars = plt.barh(sector_stats.index, sector_stats['mean'], align='center')
            
            # Add count to the labels
            labels = [f"{sector} (n={count})" for sector, count in 
                      zip(sector_stats.index, sector_stats['count'])]
            plt.yticks(range(len(labels)), labels)
            
            # Color bars based on mean value
            for i, bar in enumerate(bars):
                if sector_stats['mean'].iloc[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            # Add error bars
            plt.errorbar(sector_stats['mean'], range(len(sector_stats)), 
                         xerr=sector_stats['std'], fmt='none', color='black', alpha=0.5)
            
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            plt.title("Average Predicted Returns by Sector")
            plt.xlabel("Mean Predicted Return")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "sector_returns.png"), dpi=300)
            plt.close()
    
    elif model_type == 'graph':
        # For graph-level predictions, create a simple visualization
        plt.figure(figsize=(8, 6))
        
        # Create a gauge chart-like visualization
        prediction = predictions  # Single value for graph-level prediction
        
        # Create a simple gauge visualization
        plt.axhline(y=0.5, xmin=0.05, xmax=0.95, color='gray', alpha=0.3, linewidth=10)
        
        # Set the position based on the prediction
        # Map the prediction to [0.05, 0.95] range
        max_return = 0.1  # Assume 10% is the maximum reasonable return
        position = 0.5 + (prediction / max_return) * 0.45
        position = max(0.05, min(0.95, position))  # Clip to allowed range
        
        # Add the marker
        if prediction >= 0:
            color = 'green'
        else:
            color = 'red'
        
        plt.axhline(y=0.5, xmin=0.05, xmax=position, color=color, linewidth=10)
        
        # Add labels
        plt.text(0.05, 0.6, f"-{max_return:.1%}", fontsize=12, ha='center')
        plt.text(0.5, 0.6, "0.0%", fontsize=12, ha='center')
        plt.text(0.95, 0.6, f"+{max_return:.1%}", fontsize=12, ha='center')
        
        # Add the prediction value
        plt.text(position, 0.4, f"{prediction:.2%}", fontsize=14, 
                 ha='center', va='center', fontweight='bold')
        
        plt.text(0.5, 0.8, "Predicted Market Return", fontsize=16, 
                ha='center', fontweight='bold')
        
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "market_prediction.png"), dpi=300)
        plt.close()
    
    else:  # sector model
        # Sort sectors by predicted return
        sorted_sectors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        sectors, values = zip(*sorted_sectors)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(sectors, values, align='center')
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            if values[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title("Predicted Returns by Sector")
        plt.xlabel("Predicted Return")
        plt.ylabel("Sector")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sector_predictions.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make predictions with a trained GNN model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--lookback-days", type=int, default=20,
                       help="Number of days to use for features (default: 20)")
    
    args = parser.parse_args()
    
    predict_with_gnn(args.checkpoint, args.lookback_days)