import numpy as np
import pickle
import os
from config import DATASETS_DIR, NO_MAX_ITEMS


def generate_knapsack_trajectories(num_instances, max_items, state_dim, capacity_limit, save_path):
    """
    Generates normalized knapsack problem trajectories with weights and values scaled between [0, 1].

    Args:
        num_instances: Number of knapsack problem instances (trajectories).
        max_items: Maximum number of items in the knapsack problem.
        state_dim: Dimensionality of the item features (e.g., weight, value).
        capacity_limit: Total capacity of the knapsack.
        save_path: Filepath to save the generated trajectories.

    Returns:
        dataset: A dictionary with normalized 'states', 'actions', 'rewards', and 'returns' (RTG).
    """
    dataset = {
        'states': [],
        'actions': [],
        'rewards': [],
        'returns': []
    }

    max_weight = 100  # Max weight of items in original scale
    max_value = 100  # Max value of items in original scale

    for _ in range(num_instances):
        # Randomly generate item features (e.g., [weight, value])
        items = np.random.rand(max_items, state_dim) * 100  # Random values between 0 and 100

        # Extract weights and values, then normalize them
        item_weights = items[:, 0] / max_weight  # Normalize weights
        item_values = items[:, 1] / max_value  # Normalize values

        # Normalize knapsack capacity relative to the weight scale
        remaining_capacity = capacity_limit / max_weight

        # Initialize trajectory placeholders
        state_trajectory = []
        action_trajectory = []
        reward_trajectory = []
        return_trajectory = []

        # List to track available items
        available_items = np.arange(max_items)

        # Generate the trajectory step by step
        for step in range(max_items):
            # Current state contains remaining capacity and candidate items' (value, weight) tuples
            candidate_items = np.column_stack((item_values[available_items], item_weights[available_items]))

            # Padding to ensure fixed length
            padding_size = max_items - len(candidate_items)
            if padding_size > 0:
                candidate_items = np.pad(candidate_items, ((0, padding_size), (0, 0)), mode='constant',
                                         constant_values=0)

            # Filter items that can still be selected based on their normalized weight
            selectable_items = np.where(item_weights[available_items] <= remaining_capacity)[0]

            if len(selectable_items) == 0:
                break  # No more items can be selected due to capacity constraints

            state = np.concatenate(
                ([remaining_capacity], candidate_items.flatten()))  # [remaining_capacity, candidates]
            state_trajectory.append(state)

            # Compute value-to-weight ratio for selectable items
            value_weight_ratios = item_values[available_items[selectable_items]] / item_weights[available_items[selectable_items]]

            # Select the item with the highest value-to-weight ratio
            best_item_idx = available_items[selectable_items[np.argmax(value_weight_ratios)]]
            action_trajectory.append(best_item_idx)

            # Get the reward (normalized value) of the selected item
            reward = item_values[best_item_idx]
            reward_trajectory.append(reward)

            # Update remaining capacity with the normalized weight of the selected item
            remaining_capacity -= item_weights[best_item_idx]

            # Remove the selected item from available items
            available_items = available_items[available_items != best_item_idx]

            # Calculate Return-to-Go (RTG) as the sum of the remaining normalized values
            rtg = np.sum(item_values[available_items[item_weights[available_items] <= remaining_capacity]])
            return_trajectory.append(rtg)

        # Store the generated trajectories
        dataset['states'].append(np.array(state_trajectory))
        dataset['actions'].append(np.array(action_trajectory))
        dataset['rewards'].append(np.array(reward_trajectory))
        dataset['returns'].append(np.array(return_trajectory))


    # Save dataset to the specified path
    print(f"Before saving: len(dataset['states']) = {len(dataset['states'])}")
    print(f"len(dataset['states'][0]) = {len(dataset['states'][0])}")
    print(f"len(dataset['states'][0][0]) = {len(dataset['states'][0][0])}")

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved {num_instances} knapsack trajectories to {save_path}")
    return dataset


if __name__ == "__main__":
    # Parameters
    num_instances = 2  # Number of knapsack instances
    max_items = NO_MAX_ITEMS  # Max items in a knapsack
    state_dim = 2  # Number of features per item (e.g., weight and value)
    capacity_limit = 50
    save_path = os.path.join(DATASETS_DIR, 'knapsack_trajectories1.pkl')

    dataset = generate_knapsack_trajectories(
        num_instances=num_instances,     # Number of trajectories to generate
        max_items=max_items,          # Max number of items per trajectory
        state_dim=state_dim,           # Each item has 2 features: [weight, value]
        capacity_limit=capacity_limit,     # Total knapsack capacity
        save_path=save_path  # File to save the dataset
    )
    print("done")
