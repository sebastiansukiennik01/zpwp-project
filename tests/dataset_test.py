import os
import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch
from src.dataset import Dataset 

class TestDatasetClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the Dataset object for testing
        cls.dataset = Dataset()
        cls.dataset.load_from_csv()

    def test_load_from_csv(self):
        # Ensure data is loaded successfully
        self.assertIsNotNone(self.dataset.data)
        self.assertTrue("datetime" in self.dataset.data.columns)

    def test_save_data_to_csv(self):
        # Save data to CSV and check if the file exists
        filename = "test_data.csv"
        self.dataset.save_data_to_csv(filename)
        self.assertTrue(os.path.exists(os.path.join("real_fake", filename)))

    def test_drop_missing_values(self):
        # Create a DataFrame with missing values for testing
        dataset = Dataset()
        dataset.data = pd.DataFrame({'A': [1, 2, np.nan, 4]})
        dataset.drop_missing_values()
        self.assertFalse(dataset.data.isnull().any().any())

    def test_truncate_data(self):
        # Mocking the random number generator for deterministic testing
        with patch("numpy.random.seed"):
            self.dataset.truncate_data(k=100)
        self.assertEqual(self.dataset.data.shape[0], 100)

    def test_get_train_test(self):
        # Ensure train-test split and label extraction works
        X_train, X_test, y_train, y_test = self.dataset.get_train_test()
        self.assertTrue(X_train.shape[0] > 0)
        self.assertTrue(X_test.shape[0] > 0)
        self.assertTrue(y_train.shape[0] > 0)
        self.assertTrue(y_test.shape[0] > 0)
        
    def test_add_words_count(self):
        # Test the add_words_count method
        K = 3
        
        self.dataset.add_words_count(k=K, columns=["title", "text"])
        
        # Check if new columns are added
        columns_with_title = [col for col in self.dataset.data.columns if 'title_' in col]
        columns_with_text = [col for col in self.dataset.data.columns if 'text_' in col]
        
        self.assertTrue(len(columns_with_title) == K)
        self.assertTrue(len(columns_with_text) == K)
        

    def test_add_day_of_week(self):
        # Test the add_day_of_week method
        self.dataset.add_day_of_week()

        # Check if "weekday" column is added
        self.assertTrue("weekday" in self.dataset.data.columns)

    def test_add_month_of_year(self):
        # Test the add_month_of_year method
        self.dataset.add_month_of_year()

        # Check if "month" column is added
        self.assertTrue("month" in self.dataset.data.columns)
