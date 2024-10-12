import numpy as np
import pandas as pd
import os
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def greedy_knapsack_solver(items, capacity):
    """
    Solve the knapsack problem using a greedy heuristic (value/weight ratio).

    Args:
        items (list of dicts): Each dict contains {'item_id', 'weight', 'value'}.
        capacity (int): The capacity of the knapsack.

    Returns:
        trajectory (list of int): List of 1s and 0s indicating selected items.
    """
    # Sort items by value-to-weight ratio (highest first)
    items_sorted = sorted(items, key=lambda x: x['value'] / x['weight'], reverse=True)

    total_weight = 0
    trajectory = [0] * len(items)  # Initialize all items as not selected

    for item in items_sorted:
        if total_weight + item['weight'] <= capacity:
            # Select item
            total_weight += item['weight']
            trajectory[item['item_id'] - 1] = 1  # Mark item as selected

    return trajectory


def random_knapsack_solver(items, capacity):
    """
    Solve the knapsack problem using a random selection heuristic.

    Args:
        items (list of dicts): Each dict contains {'item_id', 'weight', 'value'}.
        capacity (int): The capacity of the knapsack.

    Returns:
        trajectory (list of int): List of 1s and 0s indicating selected items.
    """
    np.random.shuffle(items)  # Randomize item order

    total_weight = 0
    trajectory = [0] * len(items)

    for item in items:
        if total_weight + item['weight'] <= capacity:
            total_weight += item['weight']
            trajectory[item['item_id'] - 1] = 1

    return trajectory


def preprocess_knapsack_data():
    """
    Preprocess knapsack instances and generate multiple trajectories (strategies)
    for each instance.

    Returns:
    - None (saves the generated trajectories as .npy files).
    """
    folder = RAW_DATA_DIR
    save_folder = PROCESSED_DATA_DIR

    # Process each file in the folder
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path)

            # Extract features: item weight, value, and capacity
            items = df[['item_id', 'weight', 'value']].to_dict(orient='records')
            capacity = df['capacity'].values[0]

            # Generate multiple trajectories
            greedy_trajectory = greedy_knapsack_solver(items, capacity)
            random_trajectory = random_knapsack_solver(items, capacity)

            # Save each instance's data separately
            item_features = df[['weight', 'value']].values
            remaining_capacity = np.array([capacity] * len(items))
            trajectories = np.array([greedy_trajectory, random_trajectory])

            # Save individual files for each instance
            instance_id = filename.replace('.csv', '')
            np.save(os.path.join(save_folder, f'{instance_id}_item_features.npy'), item_features)
            np.save(os.path.join(save_folder, f'{instance_id}_remaining_capacity.npy'), remaining_capacity)
            np.save(os.path.join(save_folder, f'{instance_id}_trajectories.npy'), trajectories)

            print(f"Processed and saved instance {filename}")


if __name__ == "__main__":
    preprocess_knapsack_data()

