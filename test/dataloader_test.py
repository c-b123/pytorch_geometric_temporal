import unittest

import numpy as np

from torch_geometric_temporal import StaticDatasetLoader


class MyTestCase(unittest.TestCase):
    def test__get_targets_and_features_with_offset_1(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        loader.get_dataset(input_window=2, offset=3, standardize=False)

        expected_features = [[[0, 1], [1, 2], [0, 1]]]
        expected_targets = [[1, 2, 3]]

        np.testing.assert_array_equal(loader._features_train, expected_features)
        np.testing.assert_array_equal(loader._targets_train, expected_targets)

    def test__get_targets_and_features_with_offset_2(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        loader.get_dataset(input_window=3, offset=2, standardize=False)

        expected_features = [[[0, 1, 5], [1, 2, 6], [0, 1, 8]]]
        expected_targets = [[1, 2, 3]]

        np.testing.assert_array_equal(loader._features_train, expected_features)
        np.testing.assert_array_equal(loader._targets_train, expected_targets)

    def test__get_targets_and_features_with_offset_3(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        loader.get_dataset(input_window=2, offset=2, standardize=False)

        expected_features = [[[0, 1], [1, 2], [0, 1]], [[1, 5], [2, 6], [1, 8]]]
        expected_targets = [[9, 8, 7], [1, 2, 3]]

        np.testing.assert_array_equal(loader._features_train, expected_features)
        np.testing.assert_array_equal(loader._targets_train, expected_targets)

    def test__get_targets_and_features_with_offset_4(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        with self.assertRaises(Exception):
            loader.get_dataset(input_window=5, offset=5, standardize=False)

    def test_get_dataset_splitting_1(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        with self.assertRaises(Exception):
            loader.get_dataset(input_window=2, offset=3, standardize=False, val_ratio=0, test_ratio=0.2)

    def test_get_dataset_splitting_2(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        with self.assertRaises(Exception):
            loader.get_dataset(input_window=2, offset=2, standardize=False, val_ratio=0, test_ratio=0.5)

    def test_get_dataset_splitting_3(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        train, val, test = loader.get_dataset(input_window=1, offset=1, standardize=False, val_ratio=0, test_ratio=0.1)
        self.assertEqual(17, train.snapshot_count)
        self.assertEqual(1, test.snapshot_count)

    def test_get_dataset_splitting_4(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        with self.assertRaises(Exception):
            loader.get_dataset(input_window=5, offset=5, standardize=False, val_ratio=0.1, test_ratio=0.1)

    def test_get_dataset_splitting_5(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        train, val, test = loader.get_dataset(input_window=5, offset=5, standardize=False,
                                              val_ratio=0, test_ratio=0.5)
        self.assertEqual(1, train.snapshot_count)
        self.assertEqual(1, test.snapshot_count)

    def test_get_dataset_splitting_6(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        train, val, test = loader.get_dataset(input_window=2, offset=2, standardize=False,
                                              val_ratio=0.2, test_ratio=0.2)
        self.assertEqual(9, train.snapshot_count)
        self.assertEqual(1, val.snapshot_count)
        self.assertEqual(1, test.snapshot_count)

    def test_get_dataset_splitting_7(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        train, val, test = loader.get_dataset(input_window=1, offset=3, standardize=False,
                                              val_ratio=0.2, test_ratio=0.2)
        self.assertEqual(9, train.snapshot_count)
        self.assertEqual(1, val.snapshot_count)
        self.assertEqual(1, test.snapshot_count)

    def test_get_dataset_splitting_8(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        train, val, test = loader.get_dataset(input_window=3, offset=2, standardize=False,
                                              val_ratio=0.3, test_ratio=0.3)
        self.assertEqual(3, train.snapshot_count)
        self.assertEqual(3, val.snapshot_count)
        self.assertEqual(2, test.snapshot_count)

    def test_get_dataset_splitting_9(self):
        loader = StaticDatasetLoader("Resources/test_data_v3.json")
        train, val, test = loader.get_dataset(input_window=10, offset=10, standardize=False,
                                              val_ratio=0.1, test_ratio=0.1)
        print(train.snapshot_count, val.snapshot_count, test.snapshot_count)
        self.assertAlmostEqual(381, train.snapshot_count, delta=2)
        self.assertAlmostEqual(31, val.snapshot_count, delta=2)
        self.assertAlmostEqual(31, test.snapshot_count, delta=2)

    def test_training_mean_1(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        loader.get_dataset(input_window=5, offset=5, standardize=True, val_ratio=0, test_ratio=0)
        np.testing.assert_almost_equal(loader._training_mean, np.array([4.7, 4.3, 4.9]))

    def test_training_mean_2(self):
        pass

    def test_training_std_1(self):
        pass

    def test_training_std_2(self):
        pass


if __name__ == '__main__':
    unittest.main()
