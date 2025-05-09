import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

class StockGraphDataProcessor:
    """
    Process S&P 500 stock data into a graph structure for GNN models.
    Nodes are individual stocks, and edges connect stocks in the same sector.
    """
    
    def __init__(self, data_dir="data/raw", processed_dir="data/processed"):
        """
        Initialize the graph data processor.
        
        Args:
            data_dir: Directory containing raw data files
            processed_dir: Directory to save processed data
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        
        # Load raw data
        self.price_data = None
        self.sector_data = None
        self.graph_data = None
        
    def load_data(self):
        """Load price and sector data"""
        try:
            # Load price data
            price_file = os.path.join(self.data_dir, "sp500_ohlc.pkl")
            if not os.path.exists(price_file):
                raise FileNotFoundError(f"Price data file not found: {price_file}")
            self.price_data = pd.read_pickle(price_file)
            
            # Load sector data
            sector_file = os.path.join(self.data_dir, "sp500_sectors.csv")
            if not os.path.exists(sector_file):
                raise FileNotFoundError(f"Sector data file not found: {sector_file}")
            self.sector_data = pd.read_csv(sector_file)
            
            print(f"Loaded price data with {len(self.price_data.index)} trading days")
            print(f"Loaded sector data for {len(self.sector_data)} companies")
            
            return True
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_node_features(self, lookback_days=20):
        """
        Create node features from price data.
        
        Args:
            lookback_days: Number of days to use for features (default: 20)
            
        Returns:
            Dictionary mapping ticker symbols to feature tensors
        """
        if self.price_data is None or self.sector_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Align price data with sector data (keeping only stocks in both datasets)
        available_tickers = set(self.price_data.columns.levels[0]).intersection(
            set(self.sector_data['Symbol'])
        )
        
        print(f"Creating features for {len(available_tickers)} stocks")
        
        # Dictionary to store features for each ticker
        ticker_features = {}
        
        # Calculate features for each ticker
        for ticker in available_tickers:
            try:
                # Extract price data for this ticker
                ticker_price = self.price_data[ticker]
                
                # Use only the most recent lookback_days of data
                recent_data = ticker_price.iloc[-lookback_days:]
                
                # Skip tickers with insufficient data
                if len(recent_data) < lookback_days:
                    print(f"Skipping {ticker} - insufficient data")
                    continue
                
                # Create features (normalized to remove scale effects)
                # 1. Normalized close prices
                close_norm = recent_data['Close'].pct_change().fillna(0).values
                
                # 2. Volume changes
                volume_norm = recent_data['Volume'].pct_change().fillna(0).values
                
                # 3. Volatility (High-Low range)
                volatility = ((recent_data['High'] - recent_data['Low']) / 
                               recent_data['Close']).values
                
                # 4. Moving average indicators (5-day MA relative to close)
                ma5_ratio = (recent_data['Close'] / 
                              recent_data['Close'].rolling(5).mean().fillna(method='bfill')).values
                
                # Combine features
                features = np.column_stack([
                    close_norm, volume_norm, volatility, ma5_ratio
                ])
                
                # Replace NaN or Inf values with 0
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Convert to tensor and store
                ticker_features[ticker] = torch.tensor(features, dtype=torch.float)
                
            except Exception as e:
                print(f"Error creating features for {ticker}: {e}")
                continue
        
        return ticker_features
    
    def create_graph(self, node_features, visualization=False):
        """
        Create a graph where nodes are stocks and edges connect stocks in the same sector.
        
        Args:
            node_features: Dictionary mapping ticker symbols to feature tensors
            visualization: Whether to visualize the graph (default: False)
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create a mapping from ticker to numeric index
        tickers = list(node_features.keys())
        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
        
        # Filter sector data to only include tickers with features
        filtered_sector_data = self.sector_data[self.sector_data['Symbol'].isin(tickers)]
        
        # Stack node features into a single tensor
        x = torch.stack([node_features[ticker] for ticker in tickers], dim=0)
        
        # Flatten the feature tensor (batch_size, seq_len, features) -> (batch_size, seq_len * features)
        x = x.reshape(x.size(0), -1)
        
        # Create edges between stocks in the same sector
        edge_index = []
        
        # Group by sector
        for sector, group in filtered_sector_data.groupby('Sector'):
            # Get indices of stocks in this sector
            sector_indices = [ticker_to_idx[ticker] for ticker in group['Symbol'] 
                              if ticker in ticker_to_idx]
            
            # Create edges between all pairs of stocks in the sector
            for i in range(len(sector_indices)):
                for j in range(i+1, len(sector_indices)):
                    # Create bidirectional edges
                    edge_index.append([sector_indices[i], sector_indices[j]])
                    edge_index.append([sector_indices[j], sector_indices[i]])
        
        # Convert to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index)
        
        # Store tickers for reference
        graph_data.tickers = tickers
        
        # Add node-to-ticker mapping
        graph_data.idx_to_ticker = {i: ticker for i, ticker in enumerate(tickers)}
        graph_data.ticker_to_idx = ticker_to_idx
        
        # Add sector information
        sector_dict = dict(zip(filtered_sector_data['Symbol'], filtered_sector_data['Sector']))
        graph_data.sectors = [sector_dict.get(ticker, "Unknown") for ticker in tickers]
        
        print(f"Created graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
        
        # Visualize the graph if requested
        if visualization:
            self._visualize_graph(graph_data)
        
        self.graph_data = graph_data
        return graph_data
    
    def _visualize_graph(self, graph_data, max_nodes=100):
        """
        Visualize the created graph, limiting to max_nodes for clarity.
        
        Args:
            graph_data: PyTorch Geometric Data object
            max_nodes: Maximum number of nodes to show
        """
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Get unique sectors and assign colors
        unique_sectors = list(set(graph_data.sectors))
        sector_colors = {sector: plt.cm.tab10(i % 10) for i, sector in enumerate(unique_sectors)}
        
        # Add nodes with attributes
        node_colors = []
        
        # Limit to max_nodes for visualization clarity
        num_nodes = min(graph_data.num_nodes, max_nodes)
        
        for i in range(num_nodes):
            ticker = graph_data.idx_to_ticker[i]
            sector = graph_data.sectors[i]
            G.add_node(i, ticker=ticker, sector=sector)
            node_colors.append(sector_colors[sector])
        
        # Add edges (only between nodes that we're visualizing)
        for i in range(graph_data.edge_index.shape[1]):
            src, dst = graph_data.edge_index[:, i].tolist()
            if src < num_nodes and dst < num_nodes:
                G.add_edge(src, dst)
        
        # Plot
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_color=node_colors, node_size=50, alpha=0.8, linewidths=0)
        
        # Create a legend for sectors
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=sector_colors[sector], 
                                     label=sector, markersize=10) 
                          for sector in unique_sectors]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"S&P 500 Stock Graph (showing {num_nodes} of {graph_data.num_nodes} nodes)")
        
        # Save the visualization
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/stock_graph_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph visualization saved to figures/stock_graph_visualization.png")
    
    def save_graph_data(self, filename="stock_graph_data.pt"):
        """
        Save the processed graph data.
        
        Args:
            filename: Name of the file to save
        """
        if self.graph_data is None:
            raise ValueError("Graph data not created. Process data first.")
        
        output_path = os.path.join(self.processed_dir, filename)
        torch.save(self.graph_data, output_path)
        print(f"Graph data saved to {output_path}")
    
    def load_graph_data(self, filename="stock_graph_data.pt"):
        """
        Load a previously saved graph data file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            PyTorch Geometric Data object
        """
        input_path = os.path.join(self.processed_dir, filename)
        self.graph_data = torch.load(input_path)
        print(f"Loaded graph data with {self.graph_data.num_nodes} nodes and {self.graph_data.num_edges} edges")
        return self.graph_data
    
    def process(self, lookback_days=20, visualization=False):
        """
        Complete processing pipeline: load data, create features, build graph.
        
        Args:
            lookback_days: Number of days to use for features
            visualization: Whether to visualize the graph
            
        Returns:
            PyTorch Geometric Data object
        """
        if not self.load_data():
            return None
        
        node_features = self.create_node_features(lookback_days=lookback_days)
        graph_data = self.create_graph(node_features, visualization=visualization)
        self.save_graph_data()
        
        return graph_data


if __name__ == "__main__":
    # Example usage
    processor = StockGraphDataProcessor()
    graph_data = processor.process(visualization=True)
    
    if graph_data:
        print(f"Graph features shape: {graph_data.x.shape}")
        print(f"Edge index shape: {graph_data.edge_index.shape}")