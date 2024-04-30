import json
import requests
import numpy as np
import torch

from torch_geometric_temporal.signal import temporal_signal_val_split
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class StaticDatasetLoader(object):

    def __init__(self, path, colab=False):
        self.lags = None
        self.offset = None
        self.val_ratio = 0
        self.test_ratio = 0
        self._signal = None
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

    def _standardize_dataset(self):
        data = np.array(self._dataset["FX"])
        train_snapshots = int((1 - self.val_ratio - self.test_ratio) * data.shape[0])
        self.mean = np.mean(data[0:train_snapshots], axis=0)
        self.std = np.std(data[0:train_snapshots], axis=0)

        self._dataset["FX"] = (data - self.mean) / self.std

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [
            stacked_target[i: i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def _get_targets_and_features_with_offset(self):
        """
        Creates feature and target matrix. This function considers an offset. The offset is the time window between
        feature and target. Raises an exception if the dataset is too short for the specified input window and offset.
        """
        stacked_target = np.array(self._dataset["FX"])
        num_snapshots = stacked_target.shape[0] - self.lags - self.offset + 1
        if num_snapshots <= 0:
            raise Exception(
                "Feature and target vector are not specified. The input window and the offset are greater"
                " than the dataset. Check length of dataset and input window and offset.")
        self.features = [
            stacked_target[i: i + self.lags, :].T
            for i in range(num_snapshots)
        ]
        self.targets = [
            stacked_target[i + self.lags + self.offset - 1, :].T
            for i in range(num_snapshots)
        ]

    def get_dataset(self, lags: int = 4, offset: int = 1, standardize: bool = False) -> StaticGraphTemporalSignal:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
            * **offset** *(int)* - The number of time steps between features and target values.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.prepare_dataset(lags=lags, offset=offset, standardize=standardize)
        return self._signal

    def prepare_dataset(self, lags: int = 4, offset: int = 1, standardize: bool = False):
        self.lags = lags
        self.offset = offset
        if standardize:
            self._standardize_dataset()
        self._get_edges()
        self._get_edge_weights()
        if offset > 1:
            self._get_targets_and_features_with_offset()
        else:
            self._get_targets_and_features()
        self._signal = StaticGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)

    def get_train_test_split(self, lags: int = 4, offset: int = 1, standardize: bool = False, test_ratio: float = 0.2):
        self.prepare_dataset(lags=lags, offset=offset, standardize=standardize)
        return temporal_signal_split(self._signal, train_ratio=1-test_ratio)

    def get_train_val_test_split(self, lags: int = 4, offset: int = 1, standardize: bool = False,
                                 val_ratio: float = 0.1, test_ratio: float = 0.1):
        self.prepare_dataset(lags=lags, offset=offset, standardize=standardize)
        return temporal_signal_val_split(self._signal, val_ratio=val_ratio, test_ratio=test_ratio)

    def unstandardize(self, pred: torch.Tensor):
        assert pred.shape[0] == self.features[0].shape[0]
        result = np.multiply(pred.squeeze(), self.std) + self.mean
        return result


if __name__ == "__main__":
    loader = StaticDatasetLoader("Resources/test_data.json")
    # train, test = loader.get_train_test_split(lags=2, offset=3, standardize=False, test_ratio=0.2)
    data = loader.get_dataset(lags=2, offset=3, standardize=True)
    test_tensor = torch.tensor([[1], [2], [3]])
    un = loader.unstandardize(test_tensor)
    a = 54
