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
        self.layers = [DecoderBlock(model_args.layer_args) for _ in range(model_args.n_layers)]
        self.attention_mask = torch.triu(
            torch.ones(train_args.training_context, train_args.training_context) * float('-inf'),
            diagonal=1
        )
        self.pdw_to_dmodel = nn.Linear(
            in_features = train_args.data_dim,
            out_features = model_args.layer_args.d_model,
        )
        self.dmodel_to_pdw = nn.Linear(
            in_features = model_args.layer_args.d_model,
            out_features=train_args.data_dim
        )
    
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

        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_out

        x = self.norm2(x)
        ff_out = self.ff(x)
        x = x + ff_out

        return x        