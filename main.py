#!/usr/bin/env python
"""
Main entry point for the S&P 500 Market Prediction project.
"""

import os
import argparse
import logging

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

def setup_argparse():
    """Set up command line argument parsing"""
    parser = argparse.ArgumentParser(description='S&P 500 Market Prediction')
    
    # Main command selector
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data collection command
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_parser.add_argument('--update', action='store_true', help='Update the data')
    data_parser.add_argument('--days', type=int, default=365, help='Number of days to keep (default: 365)')
    
    # Training command (to be implemented)
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model', type=str, default='ggnn', help='Model type (default: ggnn)')
    
    # Prediction command (to be implemented)
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    
    return parser

def main():
    """Main function to handle command line arguments and run operations"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command == 'data' and args.update:
        logger.info(f"Running data update with {args.days}-day window")
        from scripts.common.data_collector import update_data, get_data_summary
        
        result = update_data(timeframe_days=args.days)
        if result:
            logger.info("Data update successful")
            get_data_summary()
    
    elif args.command == 'train':
        logger.info(f"Training the {args.model} model (to be implemented)")
        # Will be implemented later
        print("Training functionality will be implemented in the future")
    
    elif args.command == 'predict':
        logger.info("Making predictions (to be implemented)")
        # Will be implemented later
        print("Prediction functionality will be implemented in the future")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()