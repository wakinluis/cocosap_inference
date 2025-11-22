import numpy as np

class SequenceBuffer:
    def __init__(self, seq_len, num_features):
        self.seq_len = seq_len
        self.num_features = num_features
        self.buffer = np.zeros((seq_len, num_features), dtype=np.float32)
        self.count = 0

    def add(self, values):
        values = np.array(values, dtype=np.float32)
        if self.count < self.seq_len:
            self.buffer[self.count] = values
            self.count += 1
        else:
            # slide window
            self.buffer = np.vstack([self.buffer[1:], values])

    def ready(self):
        """Model can predict only when seq_len readings received"""
        return self.count == self.seq_len

    def get_sequence(self):
        """Return shape required by model: (1, seq_len, num_features)"""
        return self.buffer.reshape(1, self.seq_len, self.num_features)
