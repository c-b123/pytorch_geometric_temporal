import json
import requests
import numpy as np
import torch

from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class StaticClassificationDatasetLoader(object):
    # TODO merge into StaticDatasetLoader for less duplicated code

    def __init__(self, path, colab=False):
        # Input parameters
        self.input_window = None
        self.offset = None
        self.difference = None
        self.standardize = None
        self.val_ratio = None
        self.test_ratio = None
        # Computed parameters
        self._training_mean = None
        self._training_std = None
        self._edges = None
        self._edge_weights = None
        # Data
        self._raw_dataset = None
        self._features_train = None
        self._features_val = None
        self._features_test = None
        self._targets_train = None
        self._targets_val = None
        self._targets_test = None
        # Methods
        self._read_web_data(path, colab)

    def _read_web_data(self, path, colab):
        if colab:
            from google.colab import userdata
            pat = userdata.get("pat")
        else:
            from credentials import git
            pat = git["pat"]
        headers = {
            'Authorization': f'token {pat}',
            'Accept': 'application/vnd.github.v3.raw',
            'User-Agent': 'Python'
        }
        url = f'https://api.github.com/repos/c-b123/masterThesisPlay/contents/{path}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            self._raw_dataset = json.loads(response.text)
            self._fx_data = np.array(self._raw_dataset["FX"])[:, 0, :]
            self._fx_data_target = np.array(self._raw_dataset["FX"])[:, 1, :]
            print("SUCCESS Dataset loaded from GitHub")
        else:
            print(f"Failed to retrieve file: {response.status_code}")
            return None

    def _get_edges(self):
        self._edges = np.array(self._raw_dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _train_val_test_split(self):
        train_snapshots = int((1 - self.val_ratio - self.test_ratio) * self._fx_data.shape[0])
        val_snapshots = int((1 - self.test_ratio) * self._fx_data.shape[0])
        self._train = self._fx_data[0:train_snapshots]
        self._val = self._fx_data[train_snapshots:val_snapshots]
        self._test = self._fx_data[val_snapshots:]

        self._train_target = self._fx_data_target[0:train_snapshots]
        self._val_target = self._fx_data_target[train_snapshots:val_snapshots]
        self._test_target = self._fx_data_target[val_snapshots:]

    def _difference(self):
        self._train = np.diff(self._train, n=1, axis=0)
        self._val = np.diff(self._val, n=1, axis=0)
        self._test = np.diff(self._test, n=1, axis=0)

    def _standardize(self):
        self._training_mean = np.mean(self._train, axis=0)
        # self._training_std = np.std(self._train, axis=0)
        self._training_std = np.where(np.std(self._train, axis=0) == 0, 0.001, np.std(self._train, axis=0))
        self._train = (self._train - self._training_mean) / self._training_std
        self._val = (self._val - self._training_mean) / self._training_std
        self._test = (self._test - self._training_mean) / self._training_std

    def _normalize(self):
        # TODO: Implement normalization
        pass

    def _get_targets_and_features(self, dataset, dataset_targets, suffix):
        n_snapshots = dataset.shape[0] - self.input_window - self.offset + 1
        if n_snapshots <= 0:
            raise Exception(
                f"Feature and target vector of {suffix} data are not specified. The input window and the offset are "
                f"greater than the {suffix} dataset. Check length of {suffix} dataset, input window and offset.")
        features = [
            dataset[i: i + self.input_window, :].T
            for i in range(n_snapshots)
        ]
        targets = [
            dataset_targets[i + self.input_window + self.offset - 1, :].T
            for i in range(n_snapshots)
        ]
        setattr(self, f"_features{suffix}", features)
        setattr(self, f"_targets{suffix}", targets)

    def _get_targets_and_features_train(self):
        self._get_targets_and_features(self._train, self._train_target, '_train')

    def _get_targets_and_features_val(self):
        self._get_targets_and_features(self._val, self._val_target, '_val')

    def _get_targets_and_features_test(self):
        self._get_targets_and_features(self._test, self._test_target, '_test')

    def get_training_mean_and_std(self):
        return self._training_mean, self._training_std

    def get_training_min_and_max(self):
        # TODO: Implement getter for parameters of min-max normalization
        pass

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

    def destandardize(self, pred: torch.Tensor):
        # Check whether prediction array has the correct dimension
        assert pred.shape[0] == self._features_train[0].shape[0], (f"The input of dimension {pred.shape} and"
                                                                   f" the number of nodes"
                                                                   f" {self._features_train[0].shape[0]}"
                                                                   f" are not equal.")
        result = np.multiply(pred, self._training_std) + self._training_mean
        return result

    def denormalize(self, pred: torch.Tensor):
        # TODO: Implement denormalization
        pass


if __name__ == "__main__":
    loader = StaticClassificationDatasetLoader("Resources/classifier_test.json")
    data = loader.get_dataset(input_window=2, offset=3, standardize=True, val_ratio=0.1, test_ratio=0.1)

    test_tensor = torch.tensor([[1], [2], [3]])
    test_tensor_squeezed = test_tensor.squeeze()
    un = loader.destandardize(test_tensor_squeezed)
