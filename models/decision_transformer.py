import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding, which provides the transformer with information
    about the relative positions of items in a sequence. Transformers are inherently
    unaware of order, so this positional encoding is necessary for maintaining the
    sequence structure (i.e., the order of items matters in the knapsack problem).

    Args:
        d_model (int): The dimensionality of the model (i.e., the size of the embedding space).
        max_len (int): The maximum length of the input sequence (default is 500, but this can be adjusted).
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        # Initialize a tensor for the positional encodings (size: max_len x d_model)
        pe = torch.zeros(max_len, d_model)

        # Create a tensor that contains the position of each element in the sequence
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the scaling factor for the positions (used to create a sine-cosine pattern)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Apply sine and cosine to alternating dimensions of the positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd dimensions

        # Add an extra batch dimension (since the input will include batch size)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # Add the positional encodings to the input embeddings, ensuring that position information
        # is encoded in the representation the transformer receives.
        return x + self.pe[:, :x.size(1)]


class DecisionTransformer(nn.Module):
    """
    DecisionTransformer model, which processes a sequence of items and decides whether each item
    should be included in the knapsack or not, based on both item-specific information and
    global context (remaining capacity of the knapsack).

    Args:
        input_dim (int): Dimensionality of input features (e.g., item weight, value, and remaining capacity).
        embed_dim (int): Dimensionality of the embedding space (transformer input/output dimensionality).
        num_heads (int): Number of heads in the multi-head attention mechanism.
        num_layers (int): Number of transformer encoder layers (each with multi-head attention and feedforward).
        max_len (int): Maximum length of the input sequence (default is 500 items).
    """

    def __init__(self, input_dim, embed_dim, num_heads, num_layers, max_len=500):
        super(DecisionTransformer, self).__init__()

        # Embedding layer: Projects the raw input features (e.g., weight, value, and capacity) into
        # a higher-dimensional embedding space for the transformer to process.
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Positional encoding: Adds information about the position of each item in the sequence
        # to the embeddings. This helps the transformer understand the order of items.
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        # Transformer layers: Create multiple transformer encoder layers. Each encoder layer has:
        # - Multi-head attention (attends to different parts of the sequence)
        # - Feedforward layers (applies a series of transformations)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)

        # Stack multiple encoder layers to form the full transformer encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer: A fully connected layer that outputs a binary decision (1 or 0) for each item
        # in the sequence, indicating whether it should be selected or not.
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, item_features, remaining_capacity):
        """
        Forward pass of the Decision Transformer.

        Args:
            item_features (Tensor): Input features for each item in the sequence. The shape is:
                                   (batch_size, sequence_len, input_dim), where `input_dim` is typically 2
                                   (weight and value).
            remaining_capacity (Tensor): The remaining capacity of the knapsack at each step in the sequence.
                                         The shape is (batch_size, sequence_len), and it tracks how much space
                                         is left after selecting items.

        Returns:
            decision_logits (Tensor): Logits for each item, indicating how likely it is that the item should
                                      be selected (before applying a sigmoid to convert to probabilities).
                                      Shape: (batch_size, sequence_len).
        """
        # Step 1: Concatenate item features (weight, value) with remaining capacity.
        # This creates a combined representation of the item and the current state of the knapsack.
        # Input shape: (batch_size, sequence_len, input_dim + 1), where the "+1" accounts for the capacity.
        combined_input = torch.cat((item_features, remaining_capacity.unsqueeze(-1)), dim=-1)

        # Step 2: Apply the embedding layer to the combined input. This projects the low-dimensional input
        # into a higher-dimensional space (embed_dim), which is required by the transformer.
        embedded_input = self.embedding(combined_input)

        # Step 3: Add positional encodings to the embedded input. This helps the transformer understand the
        # order of the items in the sequence (which is crucial for the knapsack problem).
        embedded_input = self.positional_encoding(embedded_input)

        # Step 4: Pass the input through the transformer encoder. The transformer will process the entire
        # sequence, attending to different parts of it and capturing complex dependencies between items.
        # Output shape: (batch_size, sequence_len, embed_dim)
        transformer_output = self.transformer(embedded_input)

        # Step 5: Apply a fully connected layer to the transformer's output to produce logits for binary decisions.
        # Each logit represents the model's confidence that a particular item should be selected.
        # Output shape: (batch_size, sequence_len, 1), which we squeeze to (batch_size, sequence_len)
        decision_logits = self.fc_out(transformer_output).squeeze(-1)  # (batch_size, sequence_len)

        return decision_logits


# Example usage: This part of the code shows how the DecisionTransformer can be instantiated
# and used with some dummy data. In a real training pipeline, the input would be actual data
# representing a knapsack problem instance.

if __name__ == "__main__":
    # Create dummy data: A batch of 2 sequences, each with 5 items. Each item has 2 features: weight and value.
    # Example: item_features = [[(weight, value), (weight, value), ...], ...]
    item_features = torch.tensor([[[10, 50], [20, 60], [30, 10], [40, 40], [25, 30]],
                                  [[15, 20], [10, 40], [35, 50], [25, 10], [30, 30]]], dtype=torch.float32)

    # Remaining capacities for each sequence in the batch.
    # Example: remaining_capacity = [[capacity after selecting item 1, ...], ...]
    remaining_capacity = torch.tensor([[100, 80, 60, 40, 30], [120, 105, 85, 70, 60]], dtype=torch.float32)

    # Initialize the transformer: The input_dim is 3 (since we concatenate weight, value, and remaining capacity).
    model = DecisionTransformer(input_dim=3, embed_dim=64, num_heads=8, num_layers=6)

    # Forward pass: Run the model with the item features and remaining capacity.
    decision_logits = model(item_features, remaining_capacity)

    # The output will be the decision logits for each item in the sequence (before applying sigmoid for probabilities).
    print(decision_logits)  # Output the raw logits for each item
