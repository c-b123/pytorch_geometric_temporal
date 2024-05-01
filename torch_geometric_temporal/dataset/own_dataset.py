import json
import requests
import numpy as np
import torch

from torch_geometric_temporal.signal import temporal_signal_val_split
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class StaticDatasetLoader(object):

    def __init__(self, path, colab=False):
        # Input parameters
        self.input_window = None
        self.offset = None
        self.standardize = None
        self.val_ratio = None
        self.test_ratio = None
        # Computed parameters
        self._training_mean = None
        self._training_std = None
        self._edges = None
        self._edge_weights = None
        self._features = None
        self._targets = None
        # Data
        self._dataset = None
        self._signal = None
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
            self._dataset = json.loads(response.text)
            print("SUCCESS Dataset loaded from GitHub")
        else:
            print(f"Failed to retrieve file: {response.status_code}")
            return None

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        data_array = np.array(self._dataset["FX"])
        n_snapshots = data_array.shape[0] - self.input_window - self.offset + 1
        if n_snapshots <= 0:
            raise Exception(
                "Feature and target vector are not specified. The input window and the offset are greater"
                " than the dataset. Check length of dataset, input window and offset.")
        self._features = [
            data_array[i: i + self.input_window, :].T
            for i in range(n_snapshots)
        ]
        self._targets = [
            data_array[i + self.input_window + self.offset - 1, :].T
            for i in range(n_snapshots)
        ]

    def _standardize(self):
        data = np.array(self._dataset["FX"])
        train_snapshots = int((1 - self.val_ratio - self.test_ratio) * data.shape[0])
        self._training_mean = np.mean(data[0:train_snapshots], axis=0)
        self._training_std = np.std(data[0:train_snapshots], axis=0)
        self._dataset["FX"] = (data - self._training_mean) / self._training_std

    def _normalize(self):
        # TODO: Implement normalization
        pass

    def get_training_mean_and_std(self):
        return self._training_mean, self._training_std

    def get_training_min_and_max(self):
        # TODO: Implement getter for parameters of min-max normalization
        pass

    def get_dataset(self, input_window: int = 4, offset: int = 1, standardize: bool = True,
                    val_ratio: float = 0.1, test_ratio: float = 0.1):
        # Set parameters
        self.input_window = input_window
        self.offset = offset
        self.standardize = standardize
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Standardize if specified
        if standardize:
            self._standardize()

        # Get edges and corresponding weights
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()

        if val_ratio == 0 and test_ratio == 0:
            self._signal = StaticGraphTemporalSignal(self._edges, self._edge_weights, self._features, self._targets)
            return self._signal
        if val_ratio == 0 and not test_ratio == 0:
            self._signal = StaticGraphTemporalSignal(self._edges, self._edge_weights, self._features, self._targets)
            return temporal_signal_split(self._signal, train_ratio=1 - self.test_ratio)
        self._signal = StaticGraphTemporalSignal(self._edges, self._edge_weights, self._features, self._targets)
        return temporal_signal_val_split(self._signal, val_ratio=self.val_ratio, test_ratio=self.test_ratio)

    def destandardize(self, pred: torch.Tensor):
        # Check whether prediction array has the correct dimension
        assert pred.shape == self._features[0].shape[0], (f"The input of dimension {pred.shape} and"
                                                             f" the number of nodes {self._features[0].shape[0]}"
                                                             f" are not equal.")
        result = np.multiply(pred, self._training_std) + self._training_mean
        return result

    def denormalize(self, pred: torch.Tensor):
        # TODO: Implement denormalization
        pass


if __name__ == "__main__":
    loader = StaticDatasetLoader("Resources/test_data.json")
    data = loader.get_dataset(input_window=2, offset=3, standardize=True)
    test_tensor = torch.tensor([[1], [2], [3]])
    test_tensor_squeezed = test_tensor.squeeze()
    un = loader.destandardize(test_tensor_squeezed)
