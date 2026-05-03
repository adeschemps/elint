import matplotlib.pyplot as plt
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
    ) -> None:
        self.model = model
        self.summary_writer = summary_writer
        self.loader = loader

    def evaluate(self, step: int) -> None:
        self.model.eval()

        logger.info("Starting evaluation ...")

        losses: list[float] = []
        plot_predicted: torch.Tensor | None = None
        plot_true: torch.Tensor | None = None

        with tqdm(self.loader, total=len(self.loader)) as pbar:
            for batch in pbar:
                with torch.no_grad():
                    predicted = self.model(batch[:, :-1])
                    loss = torch.nn.functional.mse_loss(
                        input=predicted, target=batch[:, 1:], reduction="mean"
                    )
                    losses.append(loss.item())

                    if plot_predicted is None:
                        plot_predicted = predicted.cpu()
                        plot_true = batch[:, 1:].cpu()

        self.summary_writer.add_scalar(
            "Val/loss", torch.tensor(losses).mean().item(), step
        )

        for i in range(plot_predicted.shape[0]):
            fig, ax = plt.subplots()
            ax.scatter(
                plot_true[i, :, 0].numpy(), plot_true[i, :, 1].numpy(),
                label="true", alpha=0.5, s=10
            )
            ax.scatter(
                plot_predicted[i, :, 0].numpy(), plot_predicted[i, :, 1].numpy(),
                label="predicted", alpha=0.5, s=10
            )
            ax.set_xlabel("dim 0")
            ax.set_ylabel("dim 1")
            ax.legend()
            self.summary_writer.add_figure(f"Val/scatter_{i}", fig, step)
            plt.close(fig)

        self.model.train()
