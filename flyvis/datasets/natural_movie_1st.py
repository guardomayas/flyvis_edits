# flyvis/datasets/customstimuli.py

import torch
import numpy as np
import pandas as pd

from flyvis.datasets.datasets import SequenceDataset
from flyvis.datasets.render_nat_mov import RenderedNaturalMov

__all__ = ["NaturalMovie"]


class NaturalMovie(SequenceDataset):
    """
    Dataset view over RenderedNaturalMov sequences.
    Handles temporal resampling, indexing, and metadata.
    """

    dt = 1/100
    original_framerate = 60
    t_pre = 0.5
    t_post = 0.5
    n_sequences = None
    augment = False

    def __init__(self, rendered_data_config):
        super().__init__()

        # Load cached rendered dataset
        self.dir = RenderedNaturalMov(rendered_data_config)

        # Convert to torch tensors
        self.sequences = torch.tensor(self.dir.sequences[:])      # (N, T, 1, H)
        self.vel_traces = torch.tensor(self.dir.vel_traces[:])    # (N, T)

        self.n_sequences = self.sequences.shape[0]

        # metadata -> very helpful later
        self.arg_df = pd.DataFrame({
            "sequence_idx": np.arange(self.n_sequences),
            **self.dir.config,  # expand config keys
        })


    def get_item(self, key):
        """
        Returns:
            seq_resampled : (T_out, 1, H)
            vel_resampled : (T_out,)
        """

        seq = self.sequences[key]
        vel = self.vel_traces[key]

        T_raw = len(seq)

        # Determine how many time samples should remain
        idx = self.get_temporal_sample_indices(T_raw, T_raw)

        return {"lum": seq[idx], 
                "vel": vel[idx]
                }
