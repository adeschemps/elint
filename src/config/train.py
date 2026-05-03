from pathlib import Path

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class LayerArgs(StrictBaseModel):
    d_model: int = 256
    dim_ff: int = 512
    n_heads: int = 4

class ModelArgs(StrictBaseModel):
    layer_args: LayerArgs = LayerArgs()
    n_layers: int = 4


class DataArgs(StrictBaseModel):
    global_path: Path = Path(
        "/home/adeschemps/Documents/data/elint/turing-synthetic-radar-dataset/scan"
    )
    batch_size: int = 16

class OptimParams(StrictBaseModel):
    base_lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.98)

class TrainingArgs(StrictBaseModel):
    data_args: DataArgs = DataArgs()
    model_args: ModelArgs = ModelArgs()
    optim_args: OptimParams = OptimParams()
    train_ctx: int = 1024
    data_dim: int = 5
    n_epochs: int = 2
    base_lr: float = 5e-4
