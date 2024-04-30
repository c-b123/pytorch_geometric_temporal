import json
import requests
import numpy as np
import torch

from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class StaticDatasetLoader(object):

    def __init__(self, path, colab=False):
        self.lags = None
        self.offset = None
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
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

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

    def get_dataset(self, lags: int = 4, offset: int = 1, standardize: bool = True) -> StaticGraphTemporalSignal:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
            * **offset** *(int)* - The number of time steps between features and target values.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
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
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

    def unstandardize(self, pred: torch.Tensor):
        # TODO: dimensionality check
        print(self.std.shape)
        return np.multiply(pred, self.std) + self.mean


if __name__ == "__main__":
    loader = StaticDatasetLoader("Resources/test_data.json")
    dataset = loader.get_dataset(lags=2, offset=3, standardize=False)
