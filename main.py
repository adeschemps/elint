from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import PulseDataset

path = Path("/home/adeschemps/Documents/data/elint/turing-synthetic-radar-dataset/scan/train_scan")
pulse_dataset = PulseDataset(path, 1024)
pulse_dataloader = DataLoader(pulse_dataset)

test= pulse_dataset[10]
import ipdb
ipdb.set_trace()
