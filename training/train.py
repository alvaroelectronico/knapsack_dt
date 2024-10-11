import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.decision_transformer import DecisionTransformer
from data.preprocessing import preprocess_knapsack_data
import numpy as np


# Training function to train the model for a certain number of epochs
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the Decision Transformer model on the knapsack problem data.

    Args:
        model (nn.Module): The transformer model to train.
        train_loader (DataLoader): DataLoader providing batches of training data.
        criterion (nn.Module): The loss function (binary cross-entropy in this case).
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
        num_epochs (int): Number of epochs for training (default is 10).

    Returns:
        model (nn.Module): The trained model.
    """
    # Set model to training mode (important for layers like dropout, which behave differently during training)
    model.train()

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Variable to accumulate loss across the epoch
        running_loss = 0.0

        # Iterate over each batch in the DataLoader (train_loader provides batches of data)
        for i, (item_features, remaining_capacity, labels) in enumerate(train_loader):
            # Move data to the GPU if available for faster computations
            item_features = item_features.to(device)
            remaining_capacity = remaining_capacity.to(device)
            labels = labels.to(device)

            # Zero the gradients from the previous batch (this is necessary to prevent accumulation of gradients)
            optimizer.zero_grad()

            # Forward pass: compute the model's output for this batch (logits for item selection decisions)
            logits = model(item_features, remaining_capacity)

            # Apply sigmoid to convert the logits to probabilities between 0 and 1
            probs = torch.sigmoid(logits)

            # Compute the binary cross-entropy loss between the predicted probabilities and the ground-truth labels
            loss = criterion(probs, labels.float())

            # Backward pass: compute the gradients for the model's parameters
            loss.backward()

            # Update the model's parameters using the optimizer
            optimizer.step()

            # Accumulate the batch loss to track it across the epoch
            running_loss += loss.item()

        # At the end of each epoch, log the average loss for that epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    return model


# Function to load preprocessed knapsack data and prepare it for training
def load_knapsack_data(batch_size=32):
    """
    Load and preprocess knapsack problem data and create DataLoader objects.

    Args:
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
    """
    # Load preprocessed numpy arrays from files. These arrays should already be generated from preprocessing step.
    item_features = np.load('data/processed/item_features.npy')  # Shape: (num_samples, sequence_len, 2)
    remaining_capacity = np.load('data/processed/remaining_capacity.npy')  # Shape: (num_samples, sequence_len)
    labels = np.load('data/processed/labels.npy')  # Shape: (num_samples, sequence_len), 1 or 0 for each item

    # Convert numpy arrays to PyTorch tensors (this is required to work with PyTorch)
    item_features_tensor = torch.tensor(item_features, dtype=torch.float32)
    remaining_capacity_tensor = torch.tensor(remaining_capacity, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Create a TensorDataset, which holds all features and labels together, and allows easy batching
    dataset = TensorDataset(item_features_tensor, remaining_capacity_tensor, labels_tensor)

    # Create a DataLoader to easily load the data in batches
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader


if __name__ == "__main__":
    # Initialize key hyperparameters
    input_dim = 3  # item weight, value, and remaining capacity
    embed_dim = 64  # Embedding dimension size for transformer input/output
    num_heads = 8  # Number of heads in multi-head attention
    num_layers = 6  # Number of transformer layers
    batch_size = 32  # Number of samples per batch
    num_epochs = 10  # Number of epochs to train the model
    learning_rate = 1e-4  # Learning rate for the optimizer

    # Set device to GPU if available, otherwise fall back to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Decision Transformer model with the specified dimensions
    model = DecisionTransformer(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
    model = model.to(device)  # Move the model to the device (GPU or CPU)

    # Define the loss function (Binary Cross-Entropy) for binary classification
    criterion = nn.BCELoss()  # Binary cross-entropy is used for classifying each item as 1 or 0

    # Define the optimizer (Adam) to update the model's parameters during training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load training data using the custom DataLoader function
    train_loader = load_knapsack_data(batch_size=batch_size)

    # Train the model with the data, loss function, and optimizer
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # After training is complete, save the trained model's parameters to a file
    torch.save(trained_model.state_dict(), 'decision_transformer_knapsack.pth')
