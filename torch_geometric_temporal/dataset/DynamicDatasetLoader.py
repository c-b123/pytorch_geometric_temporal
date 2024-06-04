import numpy as np

from torch_geometric_temporal import DynamicGraphTemporalSignal
from torch_geometric_temporal.dataset.BaseDatasetLoader import BaseDatasetLoader


class DynamicDatasetLoader(BaseDatasetLoader):
    def __init__(self, path, colab=False):
        super().__init__(path, colab)
        self._check_dimensionality()

    def _check_dimensionality(self):
        print(self._fx_data.shape[0])
        print(len(self._raw_dataset["edges"]))
        assert self._fx_data.shape[0] == len(self._raw_dataset["edges"])
        assert self._fx_data.shape[0] == len(self._raw_dataset["edge_weights"])

    def _get_edges(self):
        edges = []
        for k, v in self._raw_dataset["edges"].items():
            edges.append(v)
        self._edges = np.array(edges, dtype=np.float32)

    def _get_edge_weights(self):
        edge_weights = []
        for k, v in self._raw_dataset["edge_weights"].items():
            edge_weights.append(v)
        self._edge_weights = np.array(edge_weights, dtype=np.float32)

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
        train_signal = DynamicGraphTemporalSignal(self._edges, self._edge_weights,
                                                 self._features_train, self._targets_train)

        val_signal = DynamicGraphTemporalSignal(self._edges, self._edge_weights, [], [])
        if val_ratio > 0:
            self._get_targets_and_features_val()
            val_signal = DynamicGraphTemporalSignal(self._edges, self._edge_weights,
                                                   self._features_val, self._targets_val)
        test_signal = DynamicGraphTemporalSignal(self._edges, self._edge_weights, [], [])
        if test_ratio > 0:
            self._get_targets_and_features_test()
            test_signal = DynamicGraphTemporalSignal(self._edges, self._edge_weights,
                                                    self._features_test, self._targets_test)

        return train_signal, val_signal, test_signal


if __name__ == '__main__':
    loader = DynamicDatasetLoader("Resources/Experiments/Dynamic/wellboat_connectivity_dynamic_ryfylke.json")

    train, val, test = loader.get_dataset(input_window=20, offset=1,
                                          difference=False, standardize=True,
                                          val_ratio=0.1, test_ratio=0.1)
