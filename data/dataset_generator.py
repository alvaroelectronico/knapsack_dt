import numpy as np
import pickle
import os
from config import DATASETS_DIR

def generate_knapsack_trajectories(num_instances, max_items, state_dim, capacity_limit, save_path):
    """
    Generates knapsack problem trajectories, including states, actions, rewards, and returns (RTG).

    Args:
        num_instances: Number of knapsack problem instances (trajectories).
        max_items: Maximum number of items in the knapsack problem.
        state_dim: Dimensionality of the item features (e.g., weight, value).
        capacity_limit: Total capacity of the knapsack.
        save_path: Filepath to save the generated trajectories.

    Returns:
        dataset: A dictionary with keys 'states', 'actions', 'rewards', and 'returns' (RTG).
    """
    dataset = {
        'states': [],
        'actions': [],
        'rewards': [],
        'returns': []
    }

    for _ in range(num_instances):
        # Randomly generate item features (e.g., [weight, value])
        items = np.random.rand(max_items, state_dim) * 100  # Random values between 0 and 100 for weight and value

        # Extract weights and values
        item_weights = items[:, 0]  # Assuming weight is the first feature
        item_values = items[:, 1]  # Assuming value is the second feature

        # Initialize remaining capacity
        remaining_capacity = capacity_limit

        # Initialize trajectory placeholders
        state_trajectory = []
        action_trajectory = []
        reward_trajectory = []
        return_trajectory = []

        # Generate the trajectory step by step
        for step in range(max_items):
            # Append the current state (remaining capacity and the item features)
            state = np.concatenate(([remaining_capacity], items.flatten()))  # [remaining_capacity, all item features]
            state_trajectory.append(state)

            # Filter items that can still be selected based on their weight
            selectable_items = np.where(item_weights <= remaining_capacity)[0]

            if len(selectable_items) == 0:
                break  # No more items can be selected due to capacity constraints


            # Compute value-to-weight ratio for selectable items
            value_weight_ratios = item_values[selectable_items] / item_weights[selectable_items]

            # Select the item with the highest value-to-weight ratio
            selected_item_idx = selectable_items[np.argmax(value_weight_ratios)]
            action_trajectory.append(selected_item_idx)

            # Get the reward (value) of the selected item
            reward = item_values[selected_item_idx]
            reward_trajectory.append(reward)

            # Update remaining capacity
            remaining_capacity -= item_weights[selected_item_idx]

            # Set the selected item's weight to infinity to mark it as already chosen
            item_weights[selected_item_idx] = float('inf')

            # Calculate Return-to-Go (RTG) as the sum of the remaining values
            rtg = np.sum(item_values[item_weights <= remaining_capacity])
            return_trajectory.append(rtg)

        # Store the generated trajectories
        dataset['states'].append(np.array(state_trajectory))
        dataset['actions'].append(np.array(action_trajectory))
        dataset['rewards'].append(np.array(reward_trajectory))
        dataset['returns'].append(np.array(return_trajectory))

    # Save dataset to the specified path
    np.save(save_path, dataset)
    print(f"Saved {num_instances} knapsack trajectories to {save_path}")

    return dataset

if __name__ == "__main__":
    # Parameters
    num_instances = 10000  # Number of knapsack instances
    max_items = 50  # Max items in a knapsack
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
