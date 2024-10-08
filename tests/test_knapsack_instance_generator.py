import unittest
import os
import pandas as pd
import json
from data.knapsack_instance_generator import KnapsackInstanceGenerator
from data.preprocessing import preprocess_knapsack_data


class TestKnapsackInstanceGenerator(unittest.TestCase):

    def setUp(self):
        """Set up the instance generator and test directories before each test."""
        self.generator = KnapsackInstanceGenerator(weight_range=(1, 100), value_range=(1, 100),
                                                   capacity_ratio=(0.5, 0.7))
        self.test_raw_folder = "data/test_raw/"
        self.test_processed_folder = "data/test_processed/"
        os.makedirs(self.test_raw_folder, exist_ok=True)
        os.makedirs(self.test_processed_folder, exist_ok=True)

    def tearDown(self):
        """Clean up the test directories after each test."""
        # Remove all files in the test directories
        for folder in [self.test_raw_folder, self.test_processed_folder]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def test_generate_instance_format(self):
        """Test that the generated instance has the correct format (items and capacity)."""
        instance = self.generator.generate_instance(num_items=10)

        # Check if 'items' is a list of dictionaries
        self.assertIsInstance(instance['items'], list)
        self.assertIsInstance(instance['items'][0], dict)

        # Check that each item has 'item_id', 'weight', and 'value'
        for item in instance['items']:
            self.assertIn('item_id', item)
            self.assertIn('weight', item)
            self.assertIn('value', item)

        # Check if the capacity is a positive integer
        self.assertIsInstance(instance['capacity'], int)
        self.assertGreater(instance['capacity'], 0)

    def test_generate_instance_weight_value_ranges(self):
        """Test that the generated weights and values are within the specified ranges."""
        instance = self.generator.generate_instance(num_items=10)

        # Check that weights and values are within the specified range
        for item in instance['items']:
            self.assertGreaterEqual(item['weight'], self.generator.weight_range[0])
            self.assertLessEqual(item['weight'], self.generator.weight_range[1])
            self.assertGreaterEqual(item['value'], self.generator.value_range[0])
            self.assertLessEqual(item['value'], self.generator.value_range[1])

    def test_save_instance(self):
        """Test saving the instance as both JSON and CSV."""
        instance = self.generator.generate_instance(num_items=10)
        filename = "test_instance"
        self.generator.save_instance(instance, filename, folder=self.test_raw_folder)

        # Check if the files were created
        json_file_path = os.path.join(self.test_raw_folder, f"{filename}.json")
        csv_file_path = os.path.join(self.test_raw_folder, f"{filename}.csv")
        self.assertTrue(os.path.exists(json_file_path))
        self.assertTrue(os.path.exists(csv_file_path))

        # Check if the JSON file can be loaded correctly
        with open(json_file_path, 'r') as f:
            loaded_json = json.load(f)
        self.assertEqual(loaded_json['capacity'], instance['capacity'])
        self.assertEqual(len(loaded_json['items']), len(instance['items']))

        # Check if the CSV file can be loaded correctly
        df = pd.read_csv(csv_file_path)
        self.assertEqual(len(df), len(instance['items']))
        self.assertIn('capacity', df.columns)

    def test_preprocessing_normalization(self):
        """Test that the preprocessing function normalizes the weights and values correctly."""
        # Generate and save an instance
        self.generator.generate_and_save_instances(num_instances=1, num_items_list=[10], folder=self.test_raw_folder)

        # Preprocess the data
        preprocess_knapsack_data(folder=self.test_raw_folder, save_folder=self.test_processed_folder)

        # Check if the processed file exists
        processed_file = os.path.join(self.test_processed_folder, "knapsack_1_10_items.csv")
        self.assertTrue(os.path.exists(processed_file))

        # Load the processed data and check normalization
        df = pd.read_csv(processed_file)
        self.assertTrue((df['weight'] <= 1.0).all())
        self.assertTrue((df['value'] <= 1.0).all())
        self.assertTrue((df['weight'] >= 0.0).all())
        self.assertTrue((df['value'] >= 0.0).all())


# Run the tests
if __name__ == '__main__':
    unittest.main()
