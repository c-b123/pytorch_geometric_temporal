import unittest

import numpy as np

from torch_geometric_temporal import StaticDatasetLoader


class MyTestCase(unittest.TestCase):
    def test__get_targets_and_features_with_offset_1(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        loader.get_dataset(lags=2, offset=3)

        expected_features = [[[0, 1], [1, 2], [0, 1]]]
        expected_targets = [[1, 2, 3]]

        np.testing.assert_array_equal(loader.features, expected_features)
        np.testing.assert_array_equal(loader.targets, expected_targets)

    def test__get_targets_and_features_with_offset_2(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        loader.get_dataset(lags=3, offset=2)

        expected_features = [[[0, 1, 5], [1, 2, 6], [0, 1, 8]]]
        expected_targets = [[1, 2, 3]]

        np.testing.assert_array_equal(loader.features, expected_features)
        np.testing.assert_array_equal(loader.targets, expected_targets)

    def test__get_targets_and_features_with_offset_3(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        loader.get_dataset(lags=2, offset=2)

        expected_features = [[[0, 1], [1, 2], [0, 1]], [[1, 5], [2, 6], [1, 8]]]
        expected_targets = [[9, 8, 7], [1, 2, 3]]

        np.testing.assert_array_equal(loader.features, expected_features)
        np.testing.assert_array_equal(loader.targets, expected_targets)


if __name__ == '__main__':
    unittest.main()
