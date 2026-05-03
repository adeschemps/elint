import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.train import DataArgs, ModelArgs, TrainingArgs
from src.data.dataset import PulseDataset
from src.model.transformer import TransformerDecoder

model_args, data_args, training_args = (
    ModelArgs(),
    DataArgs(),
    TrainingArgs()
)
train_dataset = PulseDataset(data_args.global_path, training_args.train_ctx, "train_scan")
train_loader = DataLoader(
    train_dataset,
    batch_size=data_args.batch_size,
    shuffle=True,
    drop_last=True
)

test_dataset = PulseDataset(data_args.global_path, training_args.train_ctx, "val_scan")
test_loader = DataLoader(
    test_dataset,
    batch_size=data_args.batch_size,
    shuffle=True,
    drop_last=True
)

model = TransformerDecoder(model_args=model_args, train_args=training_args)

optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=training_args.optim_args.base_lr,
    betas=training_args.optim_args.betas
)

logger.info("Starting training")

for _ in range(training_args.n_epochs):
    # training loop
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for batch in pbar:
            # model forward
            predicted = model(batch[:,:-1])
            
            # loss computation and gradient
            loss = torch.nn.functional.mse_loss(
                input=predicted, target=batch[:, 1:], reduction="mean"
            )
            model.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item()}")

    # testing loop
    model.eval()
    losses = []
    with tqdm(test_loader, total=len(test_loader)) as pbar:
        for batch in pbar:
            # model forward
            with torch.no_grad():
                predicted = model(batch[:,:-1])
                
                # loss computation and gradient
                loss = torch.nn.functional.mse_loss(
                    input=predicted, target=batch[:, 1:], reduction="mean"
                )
                losses.append(loss.item())
    mean_loss = torch.Tensor(losses).mean()

    model.train()
