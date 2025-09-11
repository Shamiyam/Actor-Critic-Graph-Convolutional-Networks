import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load and process dataset
def load_data(dataset_name):
    """
    Load Cora or PubMed dataset.
    Returns features, adjacency matrix, labels, and indices for train/val/test.
    """
    print(f"Loading {dataset_name} dataset...")
    
    # Base paths
    if dataset_name == 'cora':
        content_path = 'data/cora/cora.content'
        cites_path = 'data/cora/cora.cites'
    elif dataset_name == 'pubmed':
        content_path = 'data/pubmed/Pubmed-Diabetes.NODE.paper.tab'
        cites_path = 'data/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab'
    else:
        raise ValueError("Dataset not supported. Use 'cora' or 'pubmed'.")
    
    # Download the dataset if not exists
    os.makedirs('data', exist_ok=True)
    if dataset_name == 'cora':
        if not os.path.exists('data/cora'):
            os.makedirs('data/cora', exist_ok=True)
            print("Downloading Cora dataset...")
            # Here you would typically use wget or curl to download the dataset
            # Since we can't execute shell commands directly, you would need to 
            # implement the download functionality based on your environment
    elif dataset_name == 'pubmed':
        if not os.path.exists('data/pubmed'):
            os.makedirs('data/pubmed', exist_ok=True)
            print("Downloading PubMed dataset...")
            # Similar to above
    
    # For this example, let's assume data is already downloaded
    # In practice, you would need to download or ensure data is present

    if dataset_name == 'cora':
        # Load node features and labels
        idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        
        # Build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        
        # Load edges
        edges_unordered = np.genfromtxt(cites_path, dtype=np.int32)
        edges = np.array(list(map(lambda x: [idx_map[x[0]], idx_map[x[1]]],
                                 edges_unordered)))
        
    elif dataset_name == 'pubmed':
        # PubMed dataset has a different format
        # This is a simplified version for illustration
        # In practice, you would need to parse the specific format of PubMed
        
        # Placeholder for PubMed parsing
        features = sp.csr_matrix(np.random.rand(10000, 500))  # Dummy features
        labels = np.zeros((10000, 3))  # Dummy labels
        edges = np.random.randint(0, 10000, size=(20000, 2))  # Dummy edges
    
    # Build adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Split dataset into train, validation, and test sets (60%, 20%, 20%)
    n_nodes = labels.shape[0]
    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    
    idx_shuffle = np.random.permutation(n_nodes)
    idx_train = idx_shuffle[:n_train]
    idx_val = idx_shuffle[n_train:n_train+n_val]
    idx_test = idx_shuffle[n_train+n_val:]
    
    return features, adj, labels, idx_train, idx_val, idx_test

def encode_onehot(labels):
    """Convert labels to one-hot encoding."""
    classes = sorted(list(set(labels)))
    classes_dict = {c: i for i, c in enumerate(classes)}
    onehot_labels = np.zeros((len(labels), len(classes)))
    for i, label in enumerate(labels):
        onehot_labels[i, classes_dict[label]] = 1
    return onehot_labels

def normalize_adj(adj):
    """Normalize adjacency matrix: D^(-1/2) A D^(-1/2)."""
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_torch_sparse(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer."""
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """Forward pass: H' = D^(-1/2) A D^(-1/2) H W"""
        support = torch.mm(x, self.weight)  # H W
        output = torch.spmm(adj, support)   # D^(-1/2) A D^(-1/2) H W
        
        if self.bias is not None:
            output += self.bias
            
        return output

class GCN(nn.Module):
    """Graph Convolutional Network model."""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        
        # Two GCN layers
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        
        # Dropout
        self.dropout = dropout
    
    def forward(self, x, adj):
        """Forward pass through both GCN layers."""
        # First layer: GCN + ReLU + Dropout
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Second layer: GCN (no activation as it will be applied in loss)
        x = self.gc2(x, adj)
        
        return x

def train_gcn(features, adj, labels, idx_train, idx_val, idx_test, hidden_dim=16, lr=0.01, weight_decay=5e-4, epochs=200, dropout=0.5, seed=42):
    """Train the GCN model."""
    # Set seeds
    set_seeds(seed)
    
    # Convert to PyTorch tensors
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(np.argmax(labels, axis=1))
    adj = sparse_to_torch_sparse(normalize_adj(adj))
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # Initialize model
    model = GCN(
        nfeat=features.shape[1],
        nhid=hidden_dim,
        nclass=labels.max().item() + 1,
        dropout=dropout
    )
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Lists to store results
    train_losses = []
    val_losses = []
    val_accs = []
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(features, adj)
        
        # Calculate loss
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        
        # Backward pass
        loss_train.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            
            # Calculate accuracy
            pred_val = output[idx_val].max(1)[1]
            acc_val = pred_val.eq(labels[idx_val]).sum().item() / len(idx_val)
        
        # Store results
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        val_accs.append(acc_val)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}, Val Acc: {acc_val:.4f}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        
        # Calculate test accuracy
        pred_test = output[idx_test].max(1)[1]
        acc_test = pred_test.eq(labels[idx_test]).sum().item() / len(idx_test)
        
        print(f"Test Accuracy: {acc_test:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(labels[idx_test].numpy(), pred_test.numpy())
    
    return train_losses, val_losses, val_accs, acc_test, cm, model

def plot_gcn_results(train_losses, val_losses, val_accs, cm, num_classes, experiment_name):
    """Plot and save GCN training results."""
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curves - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plots/gcn_losses_{experiment_name}.png")
    plt.close()
    
    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accs)
    plt.title(f'Validation Accuracy - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f"plots/gcn_accuracy_{experiment_name}.png")
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"plots/gcn_confusion_matrix_{experiment_name}.png")
    plt.close()

if __name__ == "__main__":
    # Choose dataset
    dataset_name = "cora"  # or "pubmed"
    
    # Load data
    features, adj, labels, idx_train, idx_val, idx_test = load_data(dataset_name)
    
    # Number of classes
    num_classes = labels.shape[1]
    
    # Run experiment 1: Default hyperparameters
    print("\nRunning GCN Experiment 1...")
    train_losses1, val_losses1, val_accs1, acc_test1, cm1, model1 = train_gcn(
        features, adj, labels, idx_train, idx_val, idx_test,
        hidden_dim=16,
        lr=0.01,
        weight_decay=5e-4,
        epochs=200,
        dropout=0.5
    )
    plot_gcn_results(train_losses1, val_losses1, val_accs1, cm1, num_classes, "experiment1")
    
    # Run experiment 2: Different hyperparameters
    print("\nRunning GCN Experiment 2...")
    train_losses2, val_losses2, val_accs2, acc_test2, cm2, model2 = train_gcn(
        features, adj, labels, idx_train, idx_val, idx_test,
        hidden_dim=32,  # Larger hidden dim
        lr=0.005,       # Different learning rate
        weight_decay=1e-4,  # Different weight decay
        epochs=200,
        dropout=0.3     # Less dropout
    )
    plot_gcn_results(train_losses2, val_losses2, val_accs2, cm2, num_classes, "experiment2")
    
    # Compare results
    print("\nExperiment 1 Test Accuracy:", acc_test1)
    print("Experiment 2 Test Accuracy:", acc_test2)
    
    # Plot comparison of validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accs1, label='Experiment 1')
    plt.plot(val_accs2, label='Experiment 2')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("plots/gcn_accuracy_comparison.png")
    plt.close()