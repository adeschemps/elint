from pathlib import Path
from typing import Literal

import einops
import numpy as np
from h5py import File
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

# PDW dim layout: TOA, FREQ, PW, DOA, AMP
TOA_DIM, FREQ_DIM, PW_DIM, DOA_DIM, AMP_DIM = 0, 1, 2, 3, 4

_EPS = 1e-8


class PulseDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        context: int,
        subfolder: Literal["train_scan", "val_scan", "test_scan"],
    ) -> None:
        datapath = folder / subfolder
        files = [
            File(datapath / fname, "r") for fname in datapath.iterdir() if fname.suffix == ".h5"
        ]

        self.data: list[np.ndarray] = []

        logger.info(f"Loading {subfolder} ...")

        with tqdm(files, total=len(files)) as pbar:
            for file in pbar:
                data = file["data"][:]
                n_contexts = data.shape[0] // context
                cropped_data = einops.rearrange(
                    data[: n_contexts * context],
                    "(ncontexts context) dim -> ncontexts context dim",
                    context=context,
                )
                self.data.append(cropped_data)

        self.stacked_data: np.ndarray = np.concatenate(self.data, axis=0)

        logger.info("... Done.")

    def __len__(self) -> int:
        return self.stacked_data.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, float]]:
        raw = self.stacked_data[index].astype(np.float64)
        seq = np.empty_like(raw, dtype=np.float32)

        toa_min = float(raw[:, TOA_DIM].min())
        toa_max = float(raw[:, TOA_DIM].max())
        toa_range = max(toa_max - toa_min, _EPS)
        seq[:, TOA_DIM] = (raw[:, TOA_DIM] - toa_min) / toa_range

        freq_mean = float(raw[:, FREQ_DIM].mean())
        freq_std = max(float(raw[:, FREQ_DIM].std()), _EPS)
        seq[:, FREQ_DIM] = (raw[:, FREQ_DIM] - freq_mean) / freq_std

        pw_mean = float(raw[:, PW_DIM].mean())
        pw_std = max(float(raw[:, PW_DIM].std()), _EPS)
        seq[:, PW_DIM] = (raw[:, PW_DIM] - pw_mean) / pw_std

        seq[:, DOA_DIM] = raw[:, DOA_DIM] / 360.0

        amp_mean = float(raw[:, AMP_DIM].mean())
        amp_std = max(float(raw[:, AMP_DIM].std()), _EPS)
        seq[:, AMP_DIM] = (raw[:, AMP_DIM] - amp_mean) / amp_std

        stats = {
            "toa_min": toa_min,
            "toa_max": toa_max,
            "freq_mean": freq_mean,
            "freq_std": freq_std,
            "pw_mean": pw_mean,
            "pw_std": pw_std,
            "amp_mean": amp_mean,
            "amp_std": amp_std,
        }
        return seq, stats
