from flyvis import renderings_dir
from datamate import root, Directory
from flyvis.datasets.natural_mov_utils import GetNaturalMovies
from flyvis.datasets.rendering import BoxEye
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import re
import glob, os
from flyvis.datasets.datasets import SequenceDataset, MultiTaskDataset
from typing import List, Dict

__all__ = ["RenderedNaturalMov_2", "NaturalMovie"]


@root(renderings_dir)
class RenderedNaturalMov_2(Directory):

    class Config(dict):
        extent: int
        kernel_size: int
        subset_idx: list
        data_path: str = 'pano_scenes/'
        batch_idx: int = 0
        batch_size: int = 5
        traces_per_img: int = 2
        phases_per_img: int = 3
        halfLife: float = 0.2
        velStd: float = 100
        sampleFreq: int = 60
        totalTime: int = 3
        contrast_gain: float = 1.1
        fov: int = 250

    def __init__(self, config: Config):
        """
        Render natural movies with BoxEye and store them one sequence per group
        on disk under this Directory. Subsequent constructions with the same
        config will *reuse* the stored data instead of re-rendering.
        """

        # If this directory already has rendered sequences, do nothing
        if len(self) > 0:
            return

        # 1. Build natural movies (cartesian)
        gen = GetNaturalMovies(
            data_path=config.data_path,
            batch_idx=config.batch_idx,
            batch_size=config.batch_size,
            traces_per_img=config.traces_per_img,
            phases_per_img=config.phases_per_img,
            halfLife=config.halfLife,
            velStd=config.velStd,
            sampleFreq=config.sampleFreq,
            totalTime=config.totalTime,
            contrast_gain=config.contrast_gain,
            fov=config.fov,
        )

        # expose .mat filenames used in this batch
        # self.scene_filenames = np.array(gen.names)  # shape (n_imgs,)
        # expose .mat filenames used in this batch (in-memory only)
        # object.__setattr__(self, "scene_filenames", np.array(gen.names))


        # gen.all_movies: (N, T, H, fov)
        # gen.vel_traces: (n_imgs, traces_per_img, T)
        # gen.movies: (N, 3) of (img_idx, trace_idx, phase_idx)
        sequences = gen.all_movies
        vel_traces = gen.vel_traces

        # 2. BoxEye rendering (hex) and store per sequence
        receptors = BoxEye(
            extent=config.extent,
            kernel_size=config.kernel_size,
        )

        n_seq = sequences.shape[0]
        subset_idx = getattr(config, "subset_idx", []) or list(range(n_seq))

        with tqdm(total=len(subset_idx), desc="Rendering natural movies") as pbar:
            for global_idx, index in enumerate(subset_idx):
                cartesian_seq = sequences[[index]]
                lum_hex = receptors(cartesian_seq).cpu().numpy()[0]  # (T, 1, n_hex)

                img_idx, trace_idx, phase_idx = gen.movies[index]
                v = vel_traces[img_idx, trace_idx]  # (T,)

                seq_name = (
                    f"sequence_{index:05d}"
                    f"_img_{img_idx:03d}"
                    f"_trace_{trace_idx:02d}"
                    f"_phase_{phase_idx:02d}"
                )

                self[f"{seq_name}/lum"] = lum_hex   # (T, 1, n_hex)
                self[f"{seq_name}/vel"] = v         # (T,)

                pbar.update()

    def __call__(self, seq_id: int):
        """Return lum + vel for the seq_id-th rendered sequence (sorted order)."""
        group = sorted(self)[seq_id]
        data = self[group]
        return {key: data[key][:] for key in sorted(data)}  # 'lum', 'vel'

def _parse_group_name(name: str):
    """
    Parse names like 'sequence_00000_img_029_trace_00_phase_00'
    into (img_idx, trace_idx, phase_idx).
    """
    m = re.search(r"_img_(\d+)_trace_(\d+)_phase_(\d+)", name)
    if m is None:
        raise ValueError(f"Unexpected group name format: {name}")
    img_idx = int(m.group(1))
    trace_idx = int(m.group(2))
    phase_idx = int(m.group(3))
    return img_idx, trace_idx, phase_idx

# class NaturalMovie_2(SequenceDataset):
#     """
#     Dataset view over one or more RenderedNaturalMov directories.

#     Each RenderedNaturalMov corresponds to a batch of natural scenes
#     (controlled via batch_idx / batch_size in its Config).

#     This class merges all batches into a single dataset.

#     get_item returns:
#         'lum': (T_out, 1, n_hex)
#         'vel': (T_out,)
#     """

#     dt = 1 / 50
#     original_framerate = 60
#     t_pre = 0.5
#     t_post = 0.5
#     augment = False

#     def __init__(self, rendered_data_config, _init_cache: bool = False):
#         super().__init__()

#         # 1) Normalize input to a list of configs
#         if isinstance(rendered_data_config, (list, tuple)):
#             cfgs = list(rendered_data_config)
#         else:
#             cfgs = [rendered_data_config]

#         # 2) Build one RenderedNaturalMov per config (one batch each)
#         self.dirs = [RenderedNaturalMov_2(cfg) for cfg in cfgs]

#         # 3) Build a global index: dataset idx -> (dir_idx, local_idx)
#         self._index = []
#         rows = []  # metadata rows for arg_df

#         for d_ix, d in enumerate(self.dirs):
#             n_local = len(d)
#             cfg_dict = dict(d.config)

#             # group names like 'sequence_00000_img_029_trace_00_phase_00'
#             # reconstruct the batch filenames exactly like GetNaturalMovies._load_batch
#             data_path = cfg_dict["data_path"]
#             batch_idx = cfg_dict["batch_idx"]
#             batch_size = cfg_dict["batch_size"]

#             mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
#             start = batch_idx * batch_size
#             end = start + batch_size
#             batch_files = mat_files[start:end]
#             names = [os.path.basename(f) for f in batch_files]  # list of .mat names

#             group_names = sorted(d.keys())

#             for local, group_name in enumerate(group_names):
#                 global_idx = len(self._index)
#                 self._index.append((d_ix, local))

#                 img_idx, trace_idx, phase_idx = _parse_group_name(group_name)
#                 mat_file = names[img_idx]

#                 rows.append({
#                     "sequence_idx": global_idx,
#                     "dir": d_ix,
#                     "local_seq_idx": local,
#                     "group_name": group_name,
#                     "img_idx": img_idx,
#                     "trace_idx": trace_idx,
#                     "phase_idx": phase_idx,
#                     "mat_file": mat_file,
#                     **cfg_dict,
#                 })

#         self.n_sequences = len(self._index)
#         self.arg_df = pd.DataFrame(rows)

#         # optional RAM cache
#         self._cache = None
#         if _init_cache:
#             self._cache = []
#             for g_ix in range(self.n_sequences):
#                 d_ix, local_ix = self._index[g_ix]
#                 data = self.dirs[d_ix](local_ix)
#                 lum = torch.tensor(data["lum"], dtype=torch.float32)
#                 vel = torch.tensor(data["vel"], dtype=torch.float32)
#                 self._cache.append({"lum": lum, "vel": vel})

#     def __len__(self):
#         return self.n_sequences

#     def get_item(self, key: int):
#         """
#         Returns:
#             dict with
#                 'lum': (T_out, 1, n_hex)
#                 'vel': (T_out,)
#         """
#         d_ix, local_ix = self._index[key]

#         # 1) Load lum, vel for this sequence (from cache or disk)
#         if self._cache is not None:
#             lum = self._cache[key]["lum"]
#             vel = self._cache[key]["vel"]
#         else:
#             data = self.dirs[d_ix](local_ix)
#             lum = torch.tensor(data["lum"], dtype=torch.float32)
#             vel = torch.tensor(data["vel"], dtype=torch.float32)

#         T_raw = lum.shape[0]

#         # 2) Temporal resampling (provided by SequenceDataset)
#         idx = self.get_temporal_sample_indices(T_raw, T_raw)

#         return {
#             "lum": lum[idx],   # (T_out, 1, n_hex)
#             "vel": vel[idx],   # (T_out,)
#         }

def _mano_normalize_sequence(lum: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mano-style per-sequence normalization.

    For a single sequence lum with shape (T, 1, n_hex), compute a single
    scalar mean and std over *all* time × hexal entries, and z-score
    the whole sequence with those.

    Args:
        lum: (T, 1, n_hex) luminance tensor for ONE sequence
    Returns:
        lum_norm: (T, 1, n_hex) Mano-normalized luminance
    """
    #Per sequence version
    x = lum.squeeze(1)             # (T, n_hex)
    mean = x.mean()                # scalar
    std  = x.std(unbiased=False)   # scalar

    if std < eps:
        std = eps
    return (lum - mean) / std      # broadcast over (T, 1, n_hex)
    ## per column version (not used)
    # x = lum.squeeze(1)                          # (T, n_hex)
    # mean = x.mean(dim=0, keepdim=True)         # (1, n_hex)  per-column mean over time
    # std  = x.std(dim=0, keepdim=True, unbiased=False)
    # std  = torch.clamp(std, min=eps)
    # x_norm = (x - mean) / std                  # (T, n_hex)
    # return x_norm.unsqueeze(1) 


def _compute_contrast(lum: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-frame Weber contrast: (L - L_t_mean) / L_t_mean,
    where L_t_mean is mean over hexals at each time t.
    """
    x = lum.squeeze(1)                        # (T, n_hex)
    mean_t = x.mean(dim=-1, keepdim=True)    # (T, 1)

    # avoid division by zero
    mean_t_safe = mean_t.clone()
    mean_t_safe[mean_t_safe.abs() < eps] = eps

    c = (x - mean_t) / mean_t_safe           # (T, n_hex)
    return c.unsqueeze(1)                    # (T, 1, n_hex)



class NaturalMovie(MultiTaskDataset):
    """
    Dataset view over one or more RenderedNaturalMov_2 directories.

    Each RenderedNaturalMov_2 corresponds to a batch of natural scenes
    (controlled via batch_idx / batch_size in its Config).

    This class merges all batches into a single dataset.

    get_item returns:
    'raw_lum':       (T_out, 1, n_hex) raw luminance
    'lum_mano':  (T_out, 1, n_hex)     Mano-normalized luminance (per-sequence z-score)
    'contrast':  (T_out, 1, n_hex)     per-frame Weber contrast
    'vel':       (T_out,)              velocity trace (task)
    """

    # required by SequenceDataset / MultiTaskDataset
    dt = 1 / 50
    original_framerate = 60
    t_pre = 0.5
    t_post = 0.5

    # MultiTaskDataset-specific
    tasks: List[str] = ["vel"]   # supervised task(s)
    augment: bool = False        # whether temporal cropping is random

    def __init__(self, rendered_data_config, _init_cache: bool = False):
        super().__init__()

        # 1) Normalize input to a list of configs
        if isinstance(rendered_data_config, (list, tuple)):
            cfgs = list(rendered_data_config)
        else:
            cfgs = [rendered_data_config]

        # 2) Build one RenderedNaturalMov_2 per config (one batch each)
        self.dirs = [RenderedNaturalMov_2(cfg) for cfg in cfgs]

        # 3) Build a global index: dataset idx -> (dir_idx, local_idx)
        self._index = []
        rows = []  # metadata rows for arg_df

        for d_ix, d in enumerate(self.dirs):
            n_local = len(d)
            cfg_dict = dict(d.config)

            # reconstruct batch filenames like GetNaturalMovies._load_batch
            data_path = cfg_dict["data_path"]
            batch_idx = cfg_dict["batch_idx"]
            batch_size = cfg_dict["batch_size"]

            mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
            start = batch_idx * batch_size
            end = start + batch_size
            batch_files = mat_files[start:end]
            names = [os.path.basename(f) for f in batch_files]  # .mat names

            group_names = sorted(d.keys())

            for local, group_name in enumerate(group_names):
                global_idx = len(self._index)
                self._index.append((d_ix, local))

                img_idx, trace_idx, phase_idx = _parse_group_name(group_name)
                mat_file = names[img_idx]

                rows.append({
                    "sequence_idx": global_idx,
                    "dir": d_ix,
                    "local_seq_idx": local,
                    "group_name": group_name,
                    "img_idx": img_idx,
                    "trace_idx": trace_idx,
                    "phase_idx": phase_idx,
                    "mat_file": mat_file,
                    **cfg_dict,
                })

        self.n_sequences = len(self._index)
        self.arg_df = pd.DataFrame(rows)

        # optional RAM cache
        self._cache: List[Dict[str, torch.Tensor]] = None
        if _init_cache:
            self._cache = []
            for g_ix in range(self.n_sequences):
                d_ix, local_ix = self._index[g_ix]
                data = self.dirs[d_ix](local_ix)  # returns numpy arrays
                lum = torch.tensor(data["lum"], dtype=torch.float32)  # (T, 1, n_hex)
                vel = torch.tensor(data["vel"], dtype=torch.float32)  # (T,)
                self._cache.append({"lum": lum, "vel": vel})

    def __len__(self):
        return self.n_sequences

    def get_item(self, key: int):
        """
        Returns:
            dict with
                'lum':      (T_out, 1, n_hex)  raw luminance
                'contrast': (T_out, 1, n_hex)  per-frame contrast
                'vel':      (T_out,)           velocity
        """
        d_ix, local_ix = self._index[key]

        # 1) Load lum, vel for this sequence (from cache or disk)
        if self._cache is not None:
            lum = self._cache[key]["lum"]  # (T_raw, 1, n_hex)
            vel = self._cache[key]["vel"]  # (T_raw,)
        else:
            data = self.dirs[d_ix](local_ix)
            lum = torch.tensor(data["lum"], dtype=torch.float32)
            vel = torch.tensor(data["vel"], dtype=torch.float32)

        T_raw = lum.shape[0]

        # 2) Temporal resampling / cropping using SequenceDataset helper
        #    If you later want a fixed window length, pass n_frames < T_raw here.
        idx = self.get_temporal_sample_indices(T_raw, T_raw)  # uses self.augment

        lum_out = lum[idx]          # (T_out, 1, n_hex)
        vel_out = vel[idx]          # (T_out,)
        # contrast_out = _compute_contrast(lum_out)  # (T_out, 1, n_hex)

        return {
            "lum": lum_out,
            "lum_mano": _mano_normalize_sequence(lum_out),
            "contrast":  _compute_contrast(lum_out),
            "vel": vel_out,
        }
