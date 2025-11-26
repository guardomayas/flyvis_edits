import os,glob
import numpy as np
import h5py

class GetNaturalMovies:
    """
        Attributes
        ----------
        batch_imgs : np.ndarray
            Loaded and preprocessed images. Shape: (N, H, W).

        vel_traces : np.ndarray
            Velocity traces for each image and trace repetition.
            Shape: (N, traces_per_img, T).

        positions : np.ndarray
            Position (integrated velocity) traces.
            Shape: (N, traces_per_img, T).

        phase_indices : np.ndarray
            Precomputed cyclic pixel shifts for each phase.
            Shape: (phases_per_img, W).

        movies : np.ndarray
            Index table of all movie combinations.
            Shape: (num_movies, 3) containing (img_idx, trace_idx, phase_idx).

        all_movies : np.ndarray
            Rendered movies. Shape: (num_movies, T, H, fov).
    """
    def __init__(self, data_path='pano_scenes/', batch_idx=0, batch_size=5,
                 halfLife=0.2, velStd=100, sampleFreq=60, totalTime=3,
                 traces_per_img=2, phases_per_img=3, contrast_gain=1.1, fov=450):
        self.data_path = data_path
        self.batch_idx = batch_idx
        self.batch_size = batch_size
        self.traces_per_img = traces_per_img
        self.phases_per_img = phases_per_img
        self.halfLife = halfLife
        self.velStd = velStd
        self.sampleFreq = sampleFreq
        self.totalTime = totalTime
        self.numTime = int(sampleFreq * totalTime) + 1
        self.contrast_gain = contrast_gain
        self.fov = fov

        # RNG for phase indices
        self.rng = np.random.default_rng(42)

        # Load images 
        self.batch_imgs, self.names = self._load_batch()
        self.n_imgs, self.H, self.W = self.batch_imgs.shape

        # Initialize placeholders
        self.vel_traces = None
        self.positions = None
        self.phase_indices = None
        self.movies = None
        self.all_movies = None

        # Run pipeline
        self._generate_velocity_traces()
        self._generate_phase_indices()
        self._enumerate_movies()
        self._render_all_movies()

    ### METHOD DEFINITIONS ###
    # ------------------------------------------------------------------
    # Load and preprocess images
    # ------------------------------------------------------------------
    def _load_batch(self):
        mat_files = sorted(glob.glob(os.path.join(self.data_path, "*.mat")))
        start = self.batch_idx * self.batch_size
        end = start + self.batch_size
        batch_files = mat_files[start:end]

        imgs, names = [], []

        for fname in batch_files:
            with h5py.File(fname, 'r') as f:
                proj = np.array(f["projection"]).T

            img = proj / proj.max()
            names.append(os.path.basename(fname))

            if self._is_dark(img):
                print(f"[Brighten] Dark image: {fname}")
                img = self._percentile_brighten(img)

            img = (img - img.mean()) * self.contrast_gain + img.mean()
            imgs.append(img)

        imgs = np.stack(imgs)
        return np.clip(imgs, 0, 1), names


    def _is_dark(self, img, min_range=0.18, p95_threshold=0.30):
        dyn = img.max() - img.min()
        p95 = np.percentile(img, 95)
        return (dyn < min_range) or (p95 < p95_threshold)


    def _percentile_brighten(self, img, low=1, high=95):
        p_low, p_high = np.percentile(img, [low, high])
        if p_high - p_low < 1e-5:
            return img
        return np.clip((img - p_low) / (p_high - p_low), 0, 1)
    # ------------------------------------------------------------------
    # Velocity traces
    # ------------------------------------------------------------------
    def _generate_velocity_traces(self):
        self.vel_traces = np.zeros((self.n_imgs, self.traces_per_img, self.numTime))
        self.positions = np.zeros((self.n_imgs, self.traces_per_img, self.numTime))

        for i in range(self.n_imgs):
            for tr in range(self.traces_per_img):
                vel = self._vel_trace(seed=i * 100 + tr)
                pos = np.cumsum(vel) / 100.0
                pos -= pos[0]

                self.vel_traces[i, tr] = vel
                self.positions[i, tr] = pos

        print(f"[Vel] vel_traces: {self.vel_traces.shape}")
        print(f"[Vel] positions:  {self.positions.shape}")
    def _vel_trace(self, seed):
        T = self.numTime
        dt = 1.0 / self.sampleFreq
        tau = self.halfLife / np.log(2)
        alpha = np.exp(-dt / tau)

        rng = np.random.default_rng(seed)
        noise = rng.normal(0, self.velStd, size=T)

        vel = np.zeros(T)
        for t in range(1, T):
            vel[t] = alpha * vel[t-1] + np.sqrt(1 - alpha**2) * noise[t]

        return vel
    # ------------------------------------------------------------------
    # Phase indices
    # ------------------------------------------------------------------
    def _generate_phase_indices(self):
        phases_pixels = self.rng.integers(0, self.W, size=self.phases_per_img)
        self.phases_pixels = phases_pixels
        self.phase_indices = (np.arange(self.W)[None, :] - phases_pixels[:, None]) % self.W
        print(f"[Phase] Phase indices: {self.phase_indices.shape}")
    # ------------------------------------------------------------------
    # Enumerate movies
    # ------------------------------------------------------------------
    def _enumerate_movies(self):
        self.movies = np.array(np.meshgrid(
            np.arange(self.n_imgs),
            np.arange(self.traces_per_img),
            np.arange(self.phases_per_img),
            indexing='ij'
        )).reshape(3, -1).T

        self.num_movies = self.movies.shape[0]
        print(f"[Movies] Total movies: {self.num_movies}")
    # ------------------------------------------------------------------
    # Render movies
    # ------------------------------------------------------------------
    def _render_all_movies(self):
        T, H, Ww = self.numTime, self.H, self.fov

        all_movies = np.empty((self.num_movies, T, H, Ww), dtype=np.float32)

        for m, (img_idx, trace_idx, phase_idx) in enumerate(self.movies):
            all_movies[m] = self._create_movie(
                self.batch_imgs[img_idx],
                self.positions[img_idx, trace_idx],
                phase_idx,
                self.phase_indices,
                window_width=Ww
            )

        self.all_movies = all_movies
        print(f"[Render] all_movies: {self.all_movies.shape}")
    # ------------------------------------------------------------------
    # Create a single movie
    # ------------------------------------------------------------------
    def _create_movie(self, img, pos, phase_idx, phase_indices, window_width):
        shifted = img[:, phase_indices[phase_idx]]
        H, W = shifted.shape

        deg_per_pix = 360.0 / W
        pix_pos = (pos / deg_per_pix).astype(int) % W

        T = len(pix_pos)
        halfW = window_width // 2
        offsets = np.arange(-halfW, halfW)

        all_idx = (pix_pos[:, None] + offsets[None, :]) % W
        movie = shifted[:, all_idx]

        return movie.transpose(1, 0, 2)
