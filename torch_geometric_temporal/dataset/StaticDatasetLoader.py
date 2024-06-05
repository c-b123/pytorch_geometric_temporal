import numpy as np

from torch_geometric_temporal.dataset import BaseDatasetLoader
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class StaticDatasetLoader(BaseDatasetLoader):

    def __init__(self, path, colab=False):
        super().__init__(path, colab)

    def _get_edges(self):
        self._edges = np.array(self._raw_dataset["edges"], dtype=np.float32).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1], dtype=np.float32)

    def get_dataset(self, input_window: int = 4, offset: int = 1, difference: bool = False, standardize: bool = True,
                    val_ratio: float = 0, test_ratio: float = 0):
        # Set parameters
        self.input_window = input_window
        self.offset = offset
        self.difference = difference
        self.standardize = standardize
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Split dataset
        self._train_val_test_split()

        # Compute first-order difference
        if self.difference:
            self._difference()

        # Standardize if specified
        if self.standardize:
            self._standardize()

        # Get edges and corresponding weights
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features_train()
        train_signal = StaticGraphTemporalSignal(self._edges, self._edge_weights,
                                                 self._features_train, self._targets_train)

        val_signal = StaticGraphTemporalSignal(self._edges, self._edge_weights, [], [])
        if val_ratio > 0:
            self._get_targets_and_features_val()
            val_signal = StaticGraphTemporalSignal(self._edges, self._edge_weights,
                                                   self._features_val, self._targets_val)
        test_signal = StaticGraphTemporalSignal(self._edges, self._edge_weights, [], [])
        if test_ratio > 0:
            self._get_targets_and_features_test()
            test_signal = StaticGraphTemporalSignal(self._edges, self._edge_weights,
                                                    self._features_test, self._targets_test)

        return train_signal, val_signal, test_signal


if __name__ == "__main__":
    # loader = StaticDatasetLoader("Resources/test_data.json")
    # train, val, test = loader.get_dataset(input_window=2, offset=2, difference=True, standardize=True, val_ratio=0,
    #                                       test_ratio=0)
    # test_tensor = torch.tensor([[1], [2], [3]])
    # test_tensor_squeezed = test_tensor.squeeze()
    # un = loader.inverse_difference(test_tensor_squeezed, 2)

    loader = StaticDatasetLoader("Resources/Experiments/dataset_prod_area_aggregation_2012-2023.json")

    train, val, test = loader.get_dataset(input_window=20, offset=1,
                                          difference=False, standardize=True,
                                          val_ratio=0.1, test_ratio=0.1)
