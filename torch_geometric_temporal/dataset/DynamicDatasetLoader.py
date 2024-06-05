import numpy as np

from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.dataset import BaseDatasetLoader


class DynamicDatasetLoader(BaseDatasetLoader):
    def __init__(self, path, colab=False):
        super().__init__(path, colab)
        self._check_dimensionality()

    def _check_dimensionality(self):
        assert self._fx_data.shape[0] == len(self._raw_dataset["edges"])
        assert self._fx_data.shape[0] == len(self._raw_dataset["edge_weights"])

    def _get_edges(self):
        edges = []
        for k, v in self._raw_dataset["edges"].items():
            edges.append(np.array(v).T)
        self._edges = edges

    def _get_edge_weights(self):
        edge_weights = []
        for k, v in self._raw_dataset["edge_weights"].items():
            edge_weights.append(np.array(v))
        self._edge_weights = edge_weights

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

        train_snapshots_adj = self._train_snapshots - self.input_window - self.offset + 1
        val_snapshots_adj = self._val_snapshots - self.input_window - self.offset + 1

        edges_train = self._edges[0:train_snapshots_adj]
        edges_val = self._edges[self._train_snapshots:val_snapshots_adj]
        edges_test = self._edges[self._val_snapshots:- self.input_window - self.offset + 1]

        edge_weights_train = self._edge_weights[0:train_snapshots_adj]
        edge_weights_val = self._edge_weights[self._train_snapshots:val_snapshots_adj]
        edge_weights_test = self._edge_weights[self._val_snapshots:- self.input_window - self.offset + 1]

        train_signal = DynamicGraphTemporalSignal(edges_train, edge_weights_train,
                                                  self._features_train, self._targets_train)

        val_signal = DynamicGraphTemporalSignal([], [], [], [])
        if val_ratio > 0:
            self._get_targets_and_features_val()
            val_signal = DynamicGraphTemporalSignal(edges_val, edge_weights_val,
                                                    self._features_val, self._targets_val)

        test_signal = DynamicGraphTemporalSignal([], [], [], [])
        if test_ratio > 0:
            self._get_targets_and_features_test()
            test_signal = DynamicGraphTemporalSignal(edges_test, edge_weights_test,
                                                     self._features_test, self._targets_test)

        return train_signal, val_signal, test_signal


if __name__ == '__main__':
    loader = DynamicDatasetLoader("Resources/Experiments/Dynamic/dataset_dynamic_ryfylke_2012-2023.json")

    train, val, test = loader.get_dataset(input_window=20, offset=1,
                                          difference=False, standardize=True,
                                          val_ratio=0.1, test_ratio=0.1)
