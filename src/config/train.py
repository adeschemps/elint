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
    global_path = Path("/home/adeschemps/Documents/data/elint/turing-synthetic-radar-dataset/scan")
    batch_size: int = 16

class TrainingArgs(StrictBaseModel):
    data_args: DataArgs = DataArgs()
    model_args: ModelArgs = ModelArgs()
    training_context: int = 1024
    data_dim: int = 5
