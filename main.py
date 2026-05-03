import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.train import DataArgs, ModelArgs, TrainingArgs
from src.data.dataset import PulseDataset
from src.model.evaluator import Evaluator
from src.model.trainer import Trainer
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

val_dataset = PulseDataset(data_args.global_path, training_args.train_ctx, "val_scan")
val_loader = DataLoader(
    val_dataset,
    batch_size=data_args.batch_size,
    shuffle=True,
    drop_last=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerDecoder(model_args=model_args, train_args=training_args).to(device)

optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=training_args.optim_args.base_lr,
    betas=training_args.optim_args.betas
)

summary_writer = SummaryWriter(log_dir=training_args.log_dir)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    summary_writer=summary_writer,
    loader=train_loader,
    grad_accum_steps=training_args.grad_accum_steps,
    device=device,
    use_amp=training_args.mixed_precision,
)
evaluator = Evaluator(
    model=model,
    summary_writer=summary_writer,
    loader=val_loader,
    device=device,
    use_amp=training_args.mixed_precision,
)

logger.info("Starting training")

pbar = tqdm(range(training_args.n_steps))
for step in pbar:
    loss = trainer.train_step(step)
    pbar.set_postfix(loss=f"{loss:.4f}")
    if (step + 1) % training_args.eval_every == 0:
        evaluator.evaluate(step)

summary_writer.close()
