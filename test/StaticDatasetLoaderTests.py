import unittest
import torch
import numpy as np

from torch_geometric_temporal import StaticDatasetLoader


class StaticDatasetLoaderTests(unittest.TestCase):
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
        self.assertAlmostEqual(381, train.snapshot_count, delta=2)
        self.assertAlmostEqual(31, val.snapshot_count, delta=2)
        self.assertAlmostEqual(31, test.snapshot_count, delta=2)

    def test_training_mean_1(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        loader.get_dataset(input_window=5, offset=5, standardize=True, val_ratio=0, test_ratio=0)
        np.testing.assert_almost_equal(loader._training_mean, np.array([4.7, 4.3, 4.9]), decimal=3)

    def test_training_mean_2(self):
        loader = StaticDatasetLoader("Resources/test_data_v3.json")
        loader.get_dataset(input_window=5, offset=5, standardize=True, val_ratio=0, test_ratio=0.2)
        np.testing.assert_almost_equal(loader._training_mean, np.array([4.5525, 4.265, 4.4425]), decimal=3)

    def test_training_mean_3(self):
        loader = StaticDatasetLoader("Resources/test_data_v3.json")
        loader.get_dataset(input_window=2, offset=2, standardize=True, val_ratio=0.2, test_ratio=0.2)
        np.testing.assert_almost_equal(loader._training_mean, np.array([4.52333, 4.33333, 4.47667]), decimal=3)

    def test_training_std_1(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        loader.get_dataset(input_window=5, offset=5, standardize=True, val_ratio=0, test_ratio=0)
        np.testing.assert_almost_equal(loader._training_std, np.array([2.609597, 2.647640, 2.861817]), decimal=3)

    def test_training_std_2(self):
        loader = StaticDatasetLoader("Resources/test_data_v3.json")
        loader.get_dataset(input_window=5, offset=5, standardize=True, val_ratio=0, test_ratio=0.3)
        np.testing.assert_almost_equal(loader._training_std, np.array([2.940104, 2.843975, 2.926244]), decimal=3)

    def test_training_std_3(self):
        loader = StaticDatasetLoader("Resources/test_data_v3.json")
        loader.get_dataset(input_window=5, offset=5, standardize=True, val_ratio=0.2, test_ratio=0.2)
        np.testing.assert_almost_equal(loader._training_std, np.array([2.979282, 2.847611, 2.942128]), decimal=3)

    def test_standardize_1(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        loader.get_dataset(input_window=5, offset=5, standardize=True, val_ratio=0, test_ratio=0)
        train = np.array([[0.4981610824055757, -0.4910032234903236, 0.7337993857053426],
                          [-0.26824058283377167, 0.6420811384104234, 1.4326559435199548],
                          [-1.034642248073119, 0.6420811384104234, 0.7337993857053426],
                          [-0.26824058283377167, -0.4910032234903236, 0.7337993857053426],
                          [0.8813619150252494, -0.8686980107905725, 0.034942827890730485],
                          [-0.26824058283377167, -1.2463927980908216, 0.7337993857053426],
                          [0.11496024978590204, -1.2463927980908216, -0.3144854510165756],
                          [-1.8010439133124663, 1.7751655003111702, 0.034942827890730485],
                          [1.264562747644923, -1.6240875853910706, 1.4326559435199548],
                          [-1.034642248073119, 0.6420811384104234, -0.6639137299238816],
                          [1.264562747644923, -0.8686980107905725, -0.3144854510165756],
                          [-1.034642248073119, 0.6420811384104234, -0.3144854510165756],
                          [1.264562747644923, 0.6420811384104234, -1.3627702877384937],
                          [-0.6514414154534454, 1.3974707130109212, -1.3627702877384937],
                          [1.6477635802645967, 1.3974707130109212, 1.4326559435199548],
                          [-0.26824058283377167, -1.2463927980908216, -0.6639137299238816],
                          [0.4981610824055757, 1.0197759257106722, -1.0133420088311877],
                          [-1.8010439133124663, -0.4910032234903236, -1.3627702877384937],
                          [0.8813619150252494, -0.4910032234903236, -1.3627702877384937],
                          [0.11496024978590204, 0.26438635111017433, 1.4326559435199548]])
        np.testing.assert_almost_equal(loader._train, train, decimal=3)
        np.testing.assert_array_equal(loader._val, np.empty([0, 3]))
        np.testing.assert_array_equal(loader._test, np.empty([0, 3]))

    def test_standardize_2(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        loader.get_dataset(input_window=1, offset=1, standardize=True, val_ratio=0, test_ratio=0.1)
        train = np.array([[0.5365674232184475, -0.4815434123430767, 0.7802364842355226],
                          [-0.20637208585324893, 0.6019292654288461, 1.5194078903533863],
                          [-0.9493115949249454, 0.6019292654288461, 0.7802364842355226],
                          [-0.20637208585324893, -0.4815434123430767, 0.7802364842355226],
                          [0.9080371777542957, -0.8427009716003843, 0.04106507811765895],
                          [-0.20637208585324893, -1.203858530857692, 0.7802364842355226],
                          [0.16509766868259929, -1.203858530857692, -0.3285206249412729],
                          [-1.692251103996642, 1.6854019432007687, 0.04106507811765895],
                          [1.279506932290144, -1.5650160901149994, 1.5194078903533863],
                          [-0.9493115949249454, 0.6019292654288461, -0.6981063280002047],
                          [1.279506932290144, -0.8427009716003843, -0.3285206249412729],
                          [-0.9493115949249454, 0.6019292654288461, -0.3285206249412729],
                          [1.279506932290144, 0.6019292654288461, -1.4372777341180685],
                          [-0.5778418403890971, 1.3242443839434612, -1.4372777341180685],
                          [1.650976686825992, 1.3242443839434612, 1.5194078903533863],
                          [-0.20637208585324893, -1.203858530857692, -0.6981063280002047],
                          [0.5365674232184475, 0.9630868246861537, -1.0676920310591365],
                          [-1.692251103996642, -0.4815434123430767, -1.4372777341180685]])
        test = np.array([[0.9080371777542957, -0.4815434123430767, -1.4372777341180685],
                         [0.16509766868259929, 0.2407717061715385, 1.5194078903533863]])

        np.testing.assert_almost_equal(loader._train, train, decimal=3)
        np.testing.assert_array_equal(loader._val, np.empty([0, 3]))
        np.testing.assert_array_equal(loader._test, test)

    def test_standardize_3(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        loader.get_dataset(input_window=1, offset=1, standardize=True, val_ratio=0.1, test_ratio=0.1)
        train = np.array([[0.4833682445228318, -0.4402254531628119, 0.6573644635490776],
                          [-0.29002094671369905, 0.6163156344279367, 1.436463087014651],
                          [-1.0634101379502299, 0.6163156344279367, 0.6573644635490776],
                          [-0.29002094671369905, -0.4402254531628119, 0.6573644635490776],
                          [0.8700628401410971, -0.7924058156930615, -0.12173415991649586],
                          [-0.29002094671369905, -1.1445861782233109, 0.6573644635490776],
                          [0.09667364890456635, -1.1445861782233109, -0.5112834716492826],
                          [-1.8367993291867606, 1.6728567220186852, -0.12173415991649586],
                          [1.2567574357593625, -1.4967665407535604, 1.436463087014651],
                          [-1.0634101379502299, 0.6163156344279367, -0.9008327833820693],
                          [1.2567574357593625, -0.7924058156930615, -0.5112834716492826],
                          [-1.0634101379502299, 0.6163156344279367, -0.5112834716492826],
                          [1.2567574357593625, 0.6163156344279367, -1.6799314068476427],
                          [-0.6767155423319645, 1.3206763594884356, -1.6799314068476427],
                          [1.643452031377628, 1.3206763594884356, 1.436463087014651],
                          [-0.29002094671369905, -1.1445861782233109, -0.9008327833820693]])
        val = np.array([[0.4833682445228318, 0.9684959969581862, -1.2903820951148561],
                        [-1.8367993291867606, -0.4402254531628119, -1.6799314068476427]])
        test = np.array([[0.8700628401410971, -0.4402254531628119, -1.6799314068476427],
                         [0.09667364890456635, 0.26413527189768715, 1.436463087014651]])

        np.testing.assert_almost_equal(loader._train, train, decimal=3)
        np.testing.assert_almost_equal(loader._val, val, decimal=3)
        np.testing.assert_almost_equal(loader._test, test, decimal=3)

    def test_unstandardize_1(self):
        loader = StaticDatasetLoader("Resources/test_data_v2.json")
        loader.get_dataset(input_window=1, offset=1, standardize=True, val_ratio=0.1, test_ratio=0.1)
        expected = [loader._training_mean[0], loader._training_mean[1], loader._training_mean[2]]
        actual = loader.destandardize(torch.Tensor([[0], [0], [0]]).squeeze())
        np.testing.assert_almost_equal(expected, actual, decimal=3)

    def test_difference_1(self):
        loader = StaticDatasetLoader("Resources/test_data.json")
        loader.get_dataset(input_window=1, offset=1, difference=True, standardize=False, val_ratio=0, test_ratio=0)
        expected = np.array([[1, 1, 1], [4, 4, 7], [4, 2, -1], [-8, -6, -4]])
        actual = loader._train
        np.testing.assert_almost_equal(expected, actual, decimal=3)


if __name__ == '__main__':
    unittest.main()
