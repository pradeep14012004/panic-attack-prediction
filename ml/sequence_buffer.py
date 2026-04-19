"""
ml/sequence_buffer.py
─────────────────────
Rolling window buffer with 50% overlap for the AI pipeline.

Window size : 60 seconds  (SEQ_LEN feature vectors at 1Hz)
Overlap     : 50%         → new window emitted every 30 seconds
Horizon     : 2–5 minutes (HORIZON_SECS = 120s default)

Shape fed to models: (1, SEQ_LEN, N_FEATURES)
"""

import collections
import numpy as np
from ml.inference import FeatureVector

SEQ_LEN      = 60    # window size in feature vectors (1Hz → 60 seconds)
N_FEATURES   = 13    # must match FeatureVector.to_list() length
HORIZON_SECS = 120   # prediction horizon: 2 minutes ahead
OVERLAP      = 0.5   # 50% overlap → emit window every SEQ_LEN * (1-OVERLAP) steps
STEP         = int(SEQ_LEN * (1 - OVERLAP))   # = 30 steps between windows


class SequenceBuffer:
    """
    Accumulates FeatureVectors with 50% overlap sliding window.

    push(fv)      — add one feature vector
    ready()       — True when a full window is available
    get_array()   — (1, SEQ_LEN, N_FEATURES) float32 for model input
    window_ready  — True every STEP pushes once buffer is full (overlap trigger)
    """

    def __init__(self, seq_len: int = SEQ_LEN, step: int = STEP):
        self._seq_len  = seq_len
        self._step     = step
        self._buffer: collections.deque = collections.deque(maxlen=seq_len)
        self._push_count = 0

    def push(self, fv: FeatureVector):
        self._buffer.append(fv.to_list())
        self._push_count += 1

    def ready(self) -> bool:
        """Full window available."""
        return len(self._buffer) >= self._seq_len

    def window_ready(self) -> bool:
        """
        True on the first full window, then every STEP pushes after that.
        Implements 50% overlap: inference runs every 30s once warmed up.
        """
        if not self.ready():
            return False
        # First window fires at push_count == SEQ_LEN, then every STEP
        offset = self._push_count - self._seq_len
        return offset == 0 or (offset % self._step == 0)

    def get_array(self) -> np.ndarray:
        """Returns (1, SEQ_LEN, N_FEATURES) float32."""
        arr = np.array(list(self._buffer), dtype=np.float32)
        return arr[np.newaxis, ...]

    def get_flat(self) -> np.ndarray:
        """Returns (1, SEQ_LEN * N_FEATURES) for sklearn-style models."""
        return self.get_array().reshape(1, -1)

    def __len__(self) -> int:
        return len(self._buffer)
