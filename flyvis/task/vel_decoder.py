
import torch
import torch.nn as nn
import logging
from flyvis.task.decoder import ActivityDecoder
from flyvis.utils.nn_utils import n_params

__all__ = ["VelocityTemporalDecoder", "second_diff_loss"]

def second_diff_loss(conv_weight):
    # conv_weight: (C, 1, K)
    w = conv_weight.squeeze(1)          # (C, K)
    d1 = w[:, 1:] - w[:, :-1]           # first diff: (C, K-1)
    d2 = d1[:, 1:] - d1[:, :-1]         # second diff: (C, K-2)
    return (d2**2).mean()

class VelocityTemporalDecoder(ActivityDecoder):
    """
    Temporal linear decoder over spatially-pooled responses of each T4/T5 subtype.

    Steps:
      1. Pool T4/T5 over hexals -> feats: (B, T, C)
      2. Depthwise Conv1d along time per subtype (temporal kernel)
      3. LayerNorm over features at each time
      4. Linear map from C features -> scalar velocity at each time
    """
    def __init__(self, connectome, kernel_size: int = 21):
        super().__init__(connectome)

        # 1) Get output cell type names
        cell_types = connectome.output_cell_types[()].astype(str).tolist()

        # 2) Encode subtypes in canonical order
        self.t4_idx = []
        self.t5_idx = []
        for i, ct in enumerate(cell_types):
            if ct.startswith("T4"):
                self.t4_idx.append(i)
            elif ct.startswith("T5"):
                self.t5_idx.append(i)

        if not self.t4_idx or not self.t5_idx:
            raise ValueError(f"No T4/T5 channels detected in {cell_types}")

        # pooled signal has len(t4_idx)+len(t5_idx) channels
        n_features = len(self.t4_idx) + len(self.t5_idx)

        # ---- temporal kernel per subtype ----
        # shape: (C_out = C, C_in = C, kernel_size), with groups=C gives one kernel per subtype
        padding = kernel_size // 2    # centered in time; set padding=kernel_size-1 for causal+crop
        self.temp_conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=n_features,    # depthwise: each subtype gets its own kernel
            bias=False,
        )

        # ---- normalization over features at each timepoint ----
        self.norm = nn.BatchNorm1d(n_features)  # normalize last dim of (B, T, C)

        # ---- final linear readout per timepoint ----
        self.linear = nn.Linear(n_features, 1)

        self.num_parameters = n_params(self)
        logging.info(
            f"Initialized {self.__class__.__name__} with "
            f"{self.num_parameters} parameters, kernel_size={kernel_size}"
        )
    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        """
        activity: (B, T, N_cells) — Network output

        Returns:
            v_hat: (B, T) — scalar velocity per timepoint
        """
        # 1) decode DVS representation, get (B, T, C_all, n_hexals)
        self.dvs_channels.update(activity)
        x = self.dvs_channels.output  # (B, T, C_all, n_hexals)
        x = F.relu(x)

        # 2) pool each T4/T5 subtype across hexals -> (B, T, C)
        pools = []
        for idx in self.t4_idx + self.t5_idx:
            subtype_resp = x[:, :, idx, :]        # (B, T, n_hexals)
            pooled = subtype_resp.mean(dim=-1)    # (B, T)
            pools.append(pooled)

        feats = torch.stack(pools, dim=-1)        # (B, T, C)

        # 3) temporal Conv1d per subtype
        feats_perm = feats.permute(0, 2, 1)       # (B, C, T)
        feats_filt = self.temp_conv(feats_perm)   # (B, C, T)

        feats_t = feats_filt.permute(0, 2, 1)     # (B, T, C)

        # 4) BatchNorm over features (C): flatten time into batch
        B, T, C = feats_t.shape
        feats_flat = feats_t.reshape(B * T, C)    # (B*T, C)
        feats_norm_flat = self.norm(feats_flat)   # (B*T, C)
        feats_norm = feats_norm_flat.reshape(B, T, C)

        # 5) linear readout per timepoint
        v_hat = self.linear(feats_norm).squeeze(-1)  # (B, T)

        return v_hat
