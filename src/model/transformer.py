import torch
from torch import nn

from src.config.train import LayerArgs, ModelArgs, TrainingArgs


class TransformerDecoder(nn.Module):
    """Transformer decoder for PDW sequences."""
    
    def __init__(self, model_args: ModelArgs, train_args: TrainingArgs) -> None:
        """Init of decoder class.

        Args:
            model_args (ModelArgs): model arguments (d_model, input dimension, etc)
            train_args (TrainingArgs): training argument needed for model initialization
            like training context size
        
        Returns:
            None
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderBlock(model_args.layer_args) for _ in range(model_args.n_layers)]
        )
        self.pdw_to_dmodel = nn.Linear(
            in_features = train_args.data_dim,
            out_features = model_args.layer_args.d_model,
        )
        self.dmodel_to_pdw = nn.Linear(
            in_features = model_args.layer_args.d_model,
            out_features=train_args.data_dim
        )

        # registering attention mask and random positional encodings
        attention_mask = torch.triu(
            torch.ones(train_args.train_ctx - 1, train_args.train_ctx - 1) * float('-inf'),
            diagonal=1
        )
        positional_encoding = torch.rand(
            size=(1, train_args.train_ctx - 1, model_args.layer_args.d_model)
        )
        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("positional_encoding", positional_encoding)


    
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Forward of decoder model.
        
        Attention is masked so that the model only sees past pdws.

        Args:
            seq (torch.Tensor): tensor of shape (batch, seq, data_dim)
        
        Returns:
            pred_seq (torch.Tensor): tensor of shape (batch, seq, data_dim) containing
            predicted pdws.
        """
        hidden_state = self.pdw_to_dmodel(seq)
        hidden_state = hidden_state + self.positional_encoding
        for layer in self.layers:
            hidden_state = layer(hidden_state, attn_mask=self.attention_mask)
        return self.dmodel_to_pdw(hidden_state)
        


class DecoderBlock(nn.Module):
    def __init__(self, layer_args: LayerArgs) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            layer_args.d_model,
            layer_args.n_heads,
            batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(layer_args.d_model, layer_args.dim_ff),
            nn.ReLU(),
            nn.Linear(layer_args.dim_ff, layer_args.d_model)
        )
        
        self.norm1 = nn.LayerNorm(layer_args.d_model)
        self.norm2 = nn.LayerNorm(layer_args.d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None =None):
        # masked self-attention

        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = residual + ff_out

        return x        