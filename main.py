#!/usr/bin/env python
"""
Main entry point for the S&P 500 Market Prediction project.
"""

import os
import argparse
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('sp500_prediction')

logging.getLogger('scripts.train_gnn').setLevel(logging.DEBUG)

def setup_argparse():
    """Set up command line argument parsing"""
    parser = argparse.ArgumentParser(description='S&P 500 Market Prediction')
    
    # Main command selector
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data collection command
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_parser.add_argument('--update', action='store_true', help='Update the data')
    data_parser.add_argument('--days', type=int, default=365, help='Number of days to keep (default: 365)')
    data_parser.add_argument('--summary', action='store_true', help='Show data summary')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model', type=str, default='node', 
                              choices=['node', 'graph', 'sector'], help='Model type (default: node)')
    train_parser.add_argument('--gnn', type=str, default='gcn', 
                              choices=['gcn', 'gat', 'sage'], help='GNN type (default: gcn)')
    train_parser.add_argument('--lookback', type=int, default=20, help='Lookback days (default: 20)')
    train_parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon (default: 5)')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    train_parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    predict_parser.add_argument('--lookback', type=int, default=20, help='Lookback days (default: 20)')
    
    return parser

def handle_data_command(args):
    """Handle the data command"""
    from scripts.common.data_collector import update_data, get_data_summary
    
    if args.update:
        logger.info(f"Running data update with {args.days}-day window")
        result = update_data(timeframe_days=args.days)
        if result:
            logger.info("Data update successful")
        else:
            logger.error("Data update failed")
            return False
    
    if args.summary or args.update:
        summary = get_data_summary()
        if summary:
            logger.info("Data summary generated")
        else:
            logger.error("Failed to generate data summary")
            return False
    
    return True

def handle_train_command(args):
    """Handle the train command"""
    from scripts.train_gnn import train_gnn_model
    
    logger.info(f"Training {args.model} model with {args.gnn} GNN type")
    result = train_gnn_model(
        model_type=args.model,
        gnn_type=args.gnn,
        epochs=args.epochs,
        lookback_days=args.lookback,
        prediction_horizon=args.horizon,
        visualize=args.visualize
    )
    
    if result:
        logger.info("Training completed successfully")
        return True
    else:
        logger.error("Training failed")
        return False

def handle_predict_command(args):
    """Handle the predict command"""
    from scripts.predict_gnn import predict_with_gnn
    
    logger.info(f"Making predictions using checkpoint: {args.checkpoint}")
    result = predict_with_gnn(args.checkpoint, lookback_days=args.lookback)
    
    if result is not None:
        logger.info("Prediction completed successfully")
        return True
    else:
        logger.error("Prediction failed")
        return False

def main():
    """Main function to handle command line arguments and run operations"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command == 'data':
        success = handle_data_command(args)
    elif args.command == 'train':
        success = handle_train_command(args)
    elif args.command == 'predict':
        success = handle_predict_command(args)
    else:
        parser.print_help()
        return True
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())