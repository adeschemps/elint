import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        summary_writer: SummaryWriter,
        loader: DataLoader,
        n_plots: int
    ) -> None:
        self.model, self.summary_writer, self.loader = (
            model,
            summary_writer,
            loader,
    )
        self.predicted_sequences: list[torch.Tensor] = []
        self.true_sequences: list[torch.Tensor] = []
    
    def evaluate(self) -> None:
        self.model.eval()

        logger.info("Starting evaluation ...")
        
        losses = []
        with tqdm(self.loader, total=len(self.loader)) as pbar:
            for batch in pbar:
                with torch.no_grad():
                    predicted = self.model(batch[:,:-1])
                    
                    # loss computation and gradient
                    loss = torch.nn.functional.mse_loss(
                        input=predicted, target=batch[:, 1:], reduction="mean"
                    )
                    losses.append(loss.item())

        # logging mean val loss
        self.summary_writer.add_scalar("Val/loss", torch.Tensor(losses).mean())

        # logging scatterplots of true and predicted scatterplots
