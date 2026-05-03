from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _cycle(loader: DataLoader) -> Iterator[torch.Tensor]:
    """Yield batches forever, restarting the loader when it is exhausted."""
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        summary_writer: SummaryWriter,
        loader: DataLoader,
        grad_accum_steps: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.loader = loader
        self.grad_accum_steps = grad_accum_steps
        self._batch_iter = _cycle(loader)

    def train_step(self, step: int) -> float:
        self.optimizer.zero_grad()

        losses: list[float] = []
        for _ in range(self.grad_accum_steps):
            batch = next(self._batch_iter)
            predicted = self.model(batch[:, :-1])
            loss = torch.nn.functional.mse_loss(
                input=predicted, target=batch[:, 1:], reduction="mean"
            )
            (loss / self.grad_accum_steps).backward()
            losses.append(loss.item())

        self.optimizer.step()

        mean_loss = sum(losses) / len(losses)
        self.summary_writer.add_scalar("Train/loss", mean_loss, step)
        return mean_loss
