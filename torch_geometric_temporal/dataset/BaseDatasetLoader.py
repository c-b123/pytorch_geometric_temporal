import json
from abc import ABC, abstractmethod

import numpy as np
import requests
import torch


class BaseDatasetLoader(ABC):

    def __init__(self, path, colab=False):
        # Input parameters
        self.input_window = None
        self.offset = None
        self.difference = None
        self.standardize = None
        self.val_ratio = None
        self.test_ratio = None
        # Computed parameters
        self._train_snapshots = None
        self._val_snapshots = None
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
            fx_data = np.array(self._raw_dataset["FX"], dtype=np.float32)
            if len(fx_data.shape) == 3:
                self._fx_data = np.squeeze(fx_data, axis=1)
            else:
                self._fx_data = fx_data
            print("SUCCESS Dataset loaded from GitHub")
        else:
            print(f"Failed to retrieve file: {response.status_code}")
            return None

    @abstractmethod
    def _get_edges(self):
        pass

    @abstractmethod
    def _get_edge_weights(self):
        pass

    def _train_val_test_split(self):
        self._train_snapshots = int((1 - self.val_ratio - self.test_ratio) * self._fx_data.shape[0])
        self._val_snapshots = int((1 - self.test_ratio) * self._fx_data.shape[0])
        self._train = self._fx_data[0:self._train_snapshots]
        self._val = self._fx_data[self._train_snapshots:self._val_snapshots]
        self._test = self._fx_data[self._val_snapshots:]

    def _difference(self):
        # Store real values for inverse differencing
        # Assumes that data is first differenced and then standardized
        self._train_real = self._train
        self._val_real = self._val
        self._test_real = self._test
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

    def _get_targets_and_features(self, dataset, suffix):
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
            dataset[i + self.input_window + self.offset - 1, :].T
            for i in range(n_snapshots)
        ]
        setattr(self, f"_features{suffix}", features)
        setattr(self, f"_targets{suffix}", targets)

    def _get_targets_and_features_train(self):
        self._get_targets_and_features(self._train, '_train')

    def _get_targets_and_features_val(self):
        self._get_targets_and_features(self._val, '_val')

    def _get_targets_and_features_test(self):
        self._get_targets_and_features(self._test, '_test')

    def get_training_mean_and_std(self):
        return self._training_mean, self._training_std

    def get_training_min_and_max(self):
        # TODO: Implement getter for parameters of min-max normalization
        pass

    @abstractmethod
    def get_dataset(self, input_window: int = 4, offset: int = 1, difference: bool = False, standardize: bool = True,
                    val_ratio: float = 0, test_ratio: float = 0):
        pass

    def inverse_difference(self, pred: torch.Tensor, t: int, dataset="train"):
        pred_destd = self.destandardize(pred)
        torch_real = torch.from_numpy(getattr(self, f"_{dataset}_real")[t])
        return torch_real + pred_destd

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
    pass
