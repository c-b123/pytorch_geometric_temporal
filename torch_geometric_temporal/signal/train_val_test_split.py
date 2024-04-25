from typing import Union, Tuple

from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal.dynamic_graph_temporal_signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal.dynamic_graph_static_signal import DynamicGraphStaticSignal

from torch_geometric_temporal.signal.static_graph_temporal_signal_batch import StaticGraphTemporalSignalBatch
from torch_geometric_temporal.signal.dynamic_graph_temporal_signal_batch import DynamicGraphTemporalSignalBatch
from torch_geometric_temporal.signal.dynamic_graph_static_signal_batch import DynamicGraphStaticSignalBatch

from torch_geometric_temporal.signal.static_hetero_graph_temporal_signal import StaticHeteroGraphTemporalSignal
from torch_geometric_temporal.signal.dynamic_hetero_graph_temporal_signal import DynamicHeteroGraphTemporalSignal
from torch_geometric_temporal.signal.dynamic_hetero_graph_static_signal import DynamicHeteroGraphStaticSignal

from torch_geometric_temporal.signal.static_hetero_graph_temporal_signal_batch import \
    StaticHeteroGraphTemporalSignalBatch
from torch_geometric_temporal.signal.dynamic_hetero_graph_temporal_signal_batch import \
    DynamicHeteroGraphTemporalSignalBatch
from torch_geometric_temporal.signal.dynamic_hetero_graph_static_signal_batch import DynamicHeteroGraphStaticSignalBatch

Discrete_Signal = Union[
    StaticGraphTemporalSignal,
    StaticGraphTemporalSignalBatch,
    DynamicGraphTemporalSignal,
    DynamicGraphTemporalSignalBatch,
    DynamicGraphStaticSignal,
    DynamicGraphStaticSignalBatch,
    StaticHeteroGraphTemporalSignal,
    StaticHeteroGraphTemporalSignalBatch,
    DynamicHeteroGraphTemporalSignal,
    DynamicHeteroGraphTemporalSignalBatch,
    DynamicHeteroGraphStaticSignal,
    DynamicHeteroGraphStaticSignalBatch,
]


def temporal_signal_val_split(data_iterator, val_ratio: float = 0.1, test_ratio: float = 0.1) \
        -> Tuple[Discrete_Signal, Discrete_Signal, Discrete_Signal]:
    r"""Function to split a data iterator according to fixed ratios.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **val_ratio** *(float)* - Graph edge indices.
        * **test_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, val_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train, validation and
        test data iterators.
    """

    train_snapshots = int((1 - val_ratio - test_ratio) * data_iterator.snapshot_count)
    val_snapshots = train_snapshots + int(val_ratio * data_iterator.snapshot_count)

    train_iterator = data_iterator[0:train_snapshots]
    val_iterator = data_iterator[train_snapshots:val_snapshots]
    test_iterator = data_iterator[val_snapshots:]

    return train_iterator, val_iterator, test_iterator
