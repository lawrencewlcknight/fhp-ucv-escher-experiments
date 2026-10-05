"""Exact byte-coded storage for the finite Experiment 10 feature vocabulary.

This is dictionary coding, NOT numerical quantisation. Unknown float32 bit
patterns fail closed. Targets, iteration weights and continuous calibration
scalars retain float32 storage. Numeric code order preserves np.unique's
lexicographic grouping order and hence seeded minibatch choices.
"""
from __future__ import annotations

import numpy as np

STORAGE_ID = "fhp_feature_codes_v1"
SUPPORTED_ENCODER = "fhp_lossless_hand_board_v2"
CODEBOOK = np.unique(np.array(
    [k / d for d in (1, 2, 3, 5, 12, 14) for k in range(d + 1)],
    dtype=np.float32,
))
CODEBOOK.setflags(write=False)
CHUNK_ROWS = 16_384


def encode(values):
    values = np.asarray(values, dtype=np.float32)
    indices = np.searchsorted(CODEBOOK, values)
    safe = np.minimum(indices, len(CODEBOOK) - 1)
    if not np.array_equal(CODEBOOK[safe].view(np.uint32), values.view(np.uint32)):
        raise ValueError("Feature not exactly representable by the lossless FHP codebook")
    return indices.astype(np.uint8)


class EncodedFeatures:
    """Row-addressable float32 facade; only selected rows are decoded.

    No implicit ndarray conversion: that could silently materialise an entire
    replay. Calibration's non-categorical suffix is stored separately.
    """
    dtype = np.dtype(np.float32)

    def __init__(self, shape, *, coded_columns=None):
        self.shape = tuple(shape)
        self.coded_columns = self.shape[1] if coded_columns is None else int(coded_columns)
        if not 0 < self.coded_columns <= self.shape[1]:
            raise ValueError("Invalid coded feature width")
        self.codes = np.empty((self.shape[0], self.coded_columns), dtype=np.uint8)
        self.tail = np.empty((self.shape[0], self.shape[1] - self.coded_columns), dtype=np.float32)

    def __len__(self):
        return self.shape[0]

    @property
    def nbytes(self):
        return self.codes.nbytes + self.tail.nbytes

    def __array__(self, *args, **kwargs):
        raise TypeError("Select a bounded row batch before decoding replay features")

    def astype(self, dtype, *, copy=True):
        if np.dtype(dtype) == self.dtype and not copy:
            return self
        raise TypeError("Implicit full replay conversion is not supported")

    def __getitem__(self, rows):
        features = CODEBOOK[self.codes[rows]]
        if self.tail.shape[1]:
            features = np.concatenate((features, self.tail[rows]), axis=-1)
        return features

    def __setitem__(self, rows, values):
        values = np.asarray(values, dtype=np.float32)
        codes = encode(values[..., :self.coded_columns])
        self.codes[rows] = codes
        self.tail[rows] = values[..., self.coded_columns:]


def feature_state(features, size):
    if isinstance(features, EncodedFeatures):
        return {"encoding": STORAGE_ID, "codebook": CODEBOOK,
                "codes": features.codes[:size], "tail": features.tail[:size]}
    return np.asarray(features[:size])


def restore_features(destination, state, size):
    """Bounded restoration, including legacy-dense <-> coded migration."""
    if isinstance(state, dict):
        if (state.get("encoding") != STORAGE_ID or
                not np.array_equal(state["codebook"].view(np.uint32), CODEBOOK.view(np.uint32))):
            raise ValueError("Incompatible replay feature encoding")
        codes, tail = state["codes"], state["tail"]
        if (codes.dtype != np.uint8 or tail.dtype != np.float32 or
                codes.shape[0] != size or tail.shape[0] != size or
                codes.shape[1] + tail.shape[1] != destination.shape[1]):
            raise ValueError("Invalid encoded replay shape or dtype")
        for start in range(0, size, CHUNK_ROWS):
            rows = slice(start, min(start + CHUNK_ROWS, size))
            if np.any(codes[rows] >= len(CODEBOOK)):
                raise ValueError("Invalid replay feature code")
            if isinstance(destination, EncodedFeatures) and destination.coded_columns == codes.shape[1]:
                destination.codes[rows] = codes[rows]
                destination.tail[rows] = tail[rows]
            else:
                destination[rows] = np.concatenate((CODEBOOK[codes[rows]], tail[rows]), axis=1)
    else:
        if state.shape != (size, destination.shape[1]):
            raise ValueError("Invalid dense replay shape")
        for start in range(0, size, CHUNK_ROWS):
            rows = slice(start, min(start + CHUNK_ROWS, size))
            destination[rows] = state[rows]


class GroupedFeatures:
    """Minimal tensor-like interface used by the unchanged policy SGD loop."""
    def __init__(self, codes, device):
        self.codes, self.device = codes, device

    def __len__(self):
        return len(self.codes)

    def index_select(self, dim, indices):
        import torch
        if dim != 0:
            raise ValueError("Only row selection is supported")
        return torch.as_tensor(CODEBOOK[self.codes[indices.cpu().numpy()]], device=self.device)

    def full_batch(self):
        import torch
        return torch.as_tensor(CODEBOOK[self.codes], device=self.device)


def grouped_training_data(trainer, iteration):
    """Group exact byte keys; float64 sums retain original replay-row order."""
    import torch
    buffer = trainer.buffer
    size = min(buffer.cur_id, buffer.buffer_size)
    if size <= 0:
        raise ValueError("Cannot fit an average policy from an empty reservoir")
    unique, first, inverse = np.unique(buffer.infostate_buf.codes[:size], axis=0,
                                      return_index=True, return_inverse=True)
    masses = np.zeros(len(unique), dtype=np.float64)
    numerators = np.zeros((len(unique), buffer.action_size), dtype=np.float64)
    masks = buffer.q_value_mask_buf[first]
    for start in range(0, size, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, size))
        weights = np.power(np.maximum(buffer.iteration_buf[rows].astype(np.float64).reshape(-1), 0)
                           / float(iteration) * 2, trainer.gamma, dtype=np.float64)
        np.add.at(masses, inverse[rows], weights)
        np.add.at(numerators, inverse[rows], buffer.q_value_buf[rows].astype(np.float64) * weights[:, None])
        if not np.all(buffer.q_value_mask_buf[rows] == masks[inverse[rows]]):
            raise ValueError("Legal-action masks differ within an information set")
    if np.any(masses <= 0):
        raise ValueError("Grouped average-policy target has non-positive mass")
    trainer.grouped_num_rows = int(size)
    trainer.grouped_num_information_sets = len(unique)
    trainer.grouped_reduction_ratio = len(unique) / size
    trainer.grouped_target_lookup = {}
    return (GroupedFeatures(unique, trainer.device), *(torch.as_tensor(x, dtype=torch.float32, device=trainer.device)
            for x in (numerators / masses[:, None], masks, masses * (len(unique) / size))))
