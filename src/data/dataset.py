from pathlib import Path

import einops
import numpy as np
from h5py import File
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


class PulseDataset(Dataset):
    def __init__(self, folder: Path, context: int) -> None:
        
        files = [
            File(folder / fname, "r") for fname in folder.iterdir() if fname.suffix == ".h5"
        ]

        self.data: list[np.ndarray] = []
        
        logger.info("Loading data ...")
        
        with tqdm(files, total=len(files)) as pbar:
            for file in pbar:
                data = file["data"][:]
                n_contexts = data.shape[0] // context
                cropped_data = einops.rearrange(
                    data[:n_contexts * context],
                    "(ncontexts context) dim -> ncontexts context dim",
                    context=context
                )
                self.data.append(cropped_data)
        
        self.stacked_data: np.ndarray = np.concatenate(self.data, axis=0)
        
        logger.info("... Done.")

    def __len__(self) -> int:
        return self.stacked_data.shape[0]
    
    def __getitem__(self, index: int) -> np.ndarray:
        return self.stacked_data[index]

               