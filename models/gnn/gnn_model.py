import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class StockGNN(nn.Module):
    """
    Graph Neural Network model for stock market prediction.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, 
                 dropout=0.2, gnn_type='gcn', pooling='mean'):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (1 for regression, n for classification)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN layer ('gcn', 'gat', or 'sage')
            pooling: Type of graph pooling ('mean' or 'max')
        """
        super(StockGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.pooling = pooling
        
        # Create GNN layers
        self.convs = nn.ModuleList()
        
        # First layer (input to hidden)
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim))
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Additional layers (hidden to hidden)
        for _ in range(num_layers - 1):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Predictions for each node
        """
        x, edge_index = data.x, data.edge_index
        
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x


class GraphLevelStockGNN(nn.Module):
    """
    Graph-level GNN model for predicting market-wide movements.
    This model makes a single prediction for the entire graph.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, 
                 dropout=0.2, gnn_type='gcn', pooling='mean'):
        """
        Initialize the graph-level GNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN layer ('gcn', 'gat', or 'sage')
            pooling: Type of graph pooling ('mean' or 'max')
        """
        super(GraphLevelStockGNN, self).__init__()
        
        self.node_gnn = StockGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Node embeddings dimension
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type
        )
        
        self.pooling = pooling
        
        # Output layers after pooling
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, data):
        """
        Forward pass through the graph-level GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Prediction for the entire graph
        """
        # Get node embeddings from the node-level GNN
        node_embeddings = self.node_gnn(data)
        
        # Pool node embeddings to get a graph-level embedding
        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(node_embeddings, batch=None)
        elif self.pooling == 'max':
            graph_embedding = global_max_pool(node_embeddings, batch=None)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        
        # Output layers
        x = F.relu(self.fc1(graph_embedding))
        x = F.dropout(x, p=self.node_gnn.dropout, training=self.training)
        x = self.fc2(x)
        
        return x


class SectorStockGNN(nn.Module):
    """
    GNN model that makes predictions for each sector.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, 
                 dropout=0.2, gnn_type='gcn', num_sectors=11):
        """
        Initialize the sector-based GNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output per sector
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN layer ('gcn', 'gat', or 'sage')
            num_sectors: Number of sectors in the data
        """
        super(SectorStockGNN, self).__init__()
        
        self.node_gnn = StockGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Node embeddings dimension
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type
        )
        
        # Sector-specific output layers
        self.sector_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            ) for _ in range(num_sectors)
        ])
        
        self.num_sectors = num_sectors
    
    def forward(self, data):
        """
        Forward pass through the sector-based GNN.
        
        Args:
            data: PyTorch Geometric Data object with sectors attribute
            
        Returns:
            Dictionary of predictions for each sector
        """
        # Get node embeddings from the node-level GNN
        node_embeddings = self.node_gnn(data)
        
        # Create a dictionary to store sector predictions
        sector_predictions = {}
        
        # Get unique sectors in the data
        unique_sectors = set(data.sectors)
        
        # Make predictions for each sector
        for sector_idx, sector in enumerate(unique_sectors):
            if sector_idx >= self.num_sectors:
                break
                
            # Get indices of nodes in this sector
            sector_mask = [i for i, s in enumerate(data.sectors) if s == sector]
            
            if not sector_mask:
                continue
                
            # Get embeddings for this sector
            sector_embeddings = node_embeddings[sector_mask]
            
            # Pool embeddings for this sector
            sector_embedding = torch.mean(sector_embeddings, dim=0, keepdim=True)
            
            # Apply sector-specific output layer
            sector_pred = self.sector_heads[sector_idx](sector_embedding)
            
            # Store prediction
            sector_predictions[sector] = sector_pred
        
        return sector_predictions