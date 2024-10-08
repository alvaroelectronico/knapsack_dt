import numpy as np
import pandas as pd
import json
import os


class KnapsackInstanceGenerator:
    def __init__(self, weight_range=(1, 100), value_range=(1, 100), capacity_ratio=(0.5, 0.7)):
        """
        Initialize the instance generator with item weight and value ranges,
        and the capacity ratio for the knapsack.

        Parameters:
        - weight_range (tuple): Min and max range for item weights.
        - value_range (tuple): Min and max range for item values.
        - capacity_ratio (tuple): Ratio range of total weight to set knapsack capacity.
        """
        self.weight_range = weight_range
        self.value_range = value_range
        self.capacity_ratio = capacity_ratio

    def generate_instance(self, num_items):
        """
        Generate a single knapsack instance with a specified number of items.

        Parameters:
        - num_items (int): Number of items to generate.

        Returns:
        - instance (dict): Dictionary containing item weights, values, and knapsack capacity.
        """
        # Step 1: Randomly generate weights and values for the items
        weights = np.random.randint(self.weight_range[0], self.weight_range[1], size=num_items)
        values = np.random.randint(self.value_range[0], self.value_range[1], size=num_items)

        # Step 2: Calculate total weight and set knapsack capacity as a proportion of total weight
        total_weight = np.sum(weights)
        capacity = int(total_weight * np.random.uniform(self.capacity_ratio[0], self.capacity_ratio[1]))

        # Step 3: Create a DataFrame to represent the items
        df = pd.DataFrame({
            'item_id': np.arange(1, num_items + 1),
            'weight': weights,
            'value': values
        })

        # Step 4: Create a dictionary for the instance
        instance = {
            'items': df.to_dict(orient='records'),
            'capacity': capacity
        }

        return instance

    def save_instance(self, instance, filename, folder="data/raw/"):
        """
        Save a knapsack instance in both JSON and CSV formats.

        Parameters:
        - instance (dict): The knapsack instance to save.
        - filename (str): Name for the file without extension.
        - folder (str): Directory to save the instance.

        Returns:
        - None
        """
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(folder, f"{filename}.json")
        with open(json_path, 'w') as json_file:
            json.dump(instance, json_file, indent=4)

        # Convert items to DataFrame and save as CSV
        df = pd.DataFrame(instance['items'])
        df['capacity'] = instance['capacity']  # Add the capacity to the CSV
        csv_path = os.path.join(folder, f"{filename}.csv")
        df.to_csv(csv_path, index=False)

    def generate_and_save_instances(self, num_instances, num_items_list, folder="raw/"):
        """
        Generate and save multiple knapsack instances with varying numbers of items.

        Parameters:
        - num_instances (int): Number of instances to generate.
        - num_items_list (list): List of item sizes (e.g., [10, 50, 100]) for instances.
        - folder (str): Directory to save the instances.

        Returns:
        - None
        """
        for i in range(num_instances):
            for num_items in num_items_list:
                instance = self.generate_instance(num_items)
                filename = f"knapsack_{i + 1}_{num_items}_items"
                self.save_instance(instance, filename, folder)
                print(f"Saved instance {filename} to {folder}")


# Example of how to use the generator
if __name__ == "__main__":
    generator = KnapsackInstanceGenerator(weight_range=(1, 100), value_range=(1, 100), capacity_ratio=(0.5, 0.7))

    #no. instances to generate
    num_instances = 5

    # Generate and save 5 instances with varying numbers of items
    generator.generate_and_save_instances(num_instances=5, num_items_list=[10, 50, 100, 500])
