
from flyvis import renderings_dir
from datamate import root, Directory
from .natural_mov_utils import GetNaturalMovies
from flyvis.datasets.rendering import BoxEye
from tqdm import tqdm
import numpy as np

__all__ = ["RenderedNaturalMov"]

@root(renderings_dir)
class RenderedNaturalMov(Directory):

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

        # -------------------------------
        # Load data 
        # -------------------------------
        gen = GetNaturalMovies(
                            data_path=self.config.data_path,
                            batch_idx=self.config.batch_idx,
                            batch_size=self.config.batch_size,
                            traces_per_img=self.config.traces_per_img,
                            phases_per_img=self.config.phases_per_img,
                            halfLife=self.config.halfLife,
                            velStd=self.config.velStd,
                            sampleFreq=self.config.sampleFreq,
                            totalTime=self.config.totalTime,
                            contrast_gain=self.config.contrast_gain,
                            fov=self.config.fov)
        
        sequences = gen.all_movies
        vel_traces = gen.vel_traces
        
        receptors = BoxEye(
            extent=config.extent,
            kernel_size=config.kernel_size)

        # for memory-friendly rendering we can loop over individual sequences
        # and subsets of the dataset
        rendered_sequences = []
        rendered_vels = []
        subset_idx = getattr(config, "subset_idx", []) or list(range(sequences.shape[0]))
        with tqdm(total=len(subset_idx)) as pbar:
            for index in subset_idx:
                rendered_sequences.append(receptors(sequences[[index]]).cpu().numpy())
                img_idx, trace_idx, phase_idx = gen.movies[index]
                v = vel_traces[img_idx, trace_idx]   # shape (T,)
                rendered_vels.append(v[None])        # shape (1, T)
                pbar.update()

        # to join individual sequences along their first dimension
        # to obtain (n_sequences, n_frames, 1, receptors.hexals)
        rendered_sequences = np.concatenate(rendered_sequences, axis=0)
        rendered_vels = np.concatenate(rendered_vels, axis=0)
        self.sequences = rendered_sequences
        self.vel_traces = rendered_vels
        