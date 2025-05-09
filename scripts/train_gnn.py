def train_gnn_model(
    model_type='node',
    gnn_type='gcn',
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    epochs=100,
    lr=0.001,
    lookback_days=20,
    prediction_horizon=5,
    visualize=False
):
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
        lookback_days: Number of days to use for features
        prediction_horizon: Number of days ahead to predict
        visualize: Whether to visualize the graph
    """
    try:
        # Step 1: Create the processed graph data
        from scripts.common.graph_data_processor import StockGraphDataProcessor
        
        logger.info(f"Creating graph data with {lookback_days}-day features")
        processor = StockGraphDataProcessor()
        graph_data = processor.process(lookback_days=lookback_days, 
                                      visualization=visualize)
        
        if graph_data is None:
            logger.error("Failed to create graph data")
            return
        
        # Step 2: Get the raw data for creating labels
        logger.info("Loading raw data for labels")
        data_summary = get_data_summary()
        
        if data_summary is None:
            logger.error("Failed to load data summary")
            return
        
        price_data = data_summary['price_data']
        sector_data = data_summary['sector_data']
        
        # Step 3: Create the GNN trainer
        logger.info(f"Creating {model_type} GNN trainer")
        trainer = GNNTrainer(model_type=model_type)
        
        # Step 4: Prepare labels based on model type
        logger.info(f"Preparing labels with {prediction_horizon}-day prediction horizon")
        if model_type == 'node':
            labels = trainer.prepare_labels(price_data, 
                                           prediction_horizon=prediction_horizon,
                                           node_level=True)
        elif model_type == 'graph':
            labels = trainer.prepare_labels(price_data, 
                                           prediction_horizon=prediction_horizon,
                                           node_level=False)
        else:  # sector model
            labels = trainer.prepare_sector_labels(price_data, graph_data,
                                                 prediction_horizon=prediction_horizon)
        
        # Step 5: Create the model
        input_dim = graph_data.x.size(1)  # Feature dimension
        num_sectors = len(set(graph_data.sectors))
        
        logger.info(f"Creating model with input dim: {input_dim}, sectors: {num_sectors}")
        trainer.create_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # Regression model predicting future returns
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            num_sectors=num_sectors
        )
        
        # Step 6: Train the model
        logger.info(f"Training model for {epochs} epochs")
        history = trainer.train(
            graph_data=graph_data,
            labels=labels,
            epochs=epochs,
            lr=lr,
            weight_decay=1e-5,
            validation_split=0.2,
            early_stopping=10
        )
        
        # Step 7: Plot and save training history
        logger.info("Creating training history plot")
        os.makedirs("figures", exist_ok=True)
        trainer.plot_training_history(
            history,
            save_path=f"figures/{model_type}_training_history.png"
        )
        
        # Step 8: Make predictions
        logger.info("Making predictions")
        predictions = trainer.predict(graph_data)
        
        if model_type == 'node':
            # Sort tickers by predicted return (descending)
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Print top 10 positive and negative predictions
            print("\nTop 10 stocks with highest predicted returns:")
            for ticker, pred in sorted_preds[:10]:
                print(f"{ticker}: {pred:.4f}")
            
            print("\nTop 10 stocks with lowest predicted returns:")
            for ticker, pred in sorted_preds[-10:]:
                print(f"{ticker}: {pred:.4f}")
                
        elif model_type == 'graph':
            print(f"\nPredicted market return: {predictions:.4f}")
            
        else:  # sector model
            # Sort sectors by predicted return (descending)
            sorted_sectors = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            print("\nPredicted sector returns (descending):")
            for sector, pred in sorted_sectors:
                print(f"{sector}: {pred:.4f}")
        
        logger.info("GNN training and evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Error in GNN training: {str(e)}")
        return False