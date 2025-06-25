import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from tqdm import tqdm
import os
import wandb
from argparse import ArgumentParser

from d3pm_runner import D3PM, D3PMLightning
from dit import DiT_Llama


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64, num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # Download data if needed
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == 'fit' or stage is None:
            self.cifar_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            # Use a small subset for validation
            train_size = int(0.9 * len(self.cifar_train))
            val_size = len(self.cifar_train) - train_size
            self.cifar_train, self.cifar_val = torch.utils.data.random_split(
                self.cifar_train, [train_size, val_size]
            )

        # Assign test dataset
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class CIFAR10D3PMLightning(D3PMLightning):
    def __init__(
        self,
        n_channel: int = 3,
        N: int = 8,
        n_T: int = 1000,
        num_classes: int = 8,
        forward_type: str = "uniform",
        hybrid_loss_coeff: float = 0.0,
        lr: float = 2e-5,
    ) -> None:
        super(D3PMLightning, self).__init__()  # Skip D3PMLightning's init to customize the model
        self.save_hyperparameters()
        self.x0_model = DiT_Llama(n_channel, N, dim=1024)
        self.d3pm = D3PM(
            self.x0_model, n_T, num_classes, forward_type, hybrid_loss_coeff
        )
        self.N = N
        self.lr = lr

    def on_validation_epoch_end(self):
        if self.global_step == 0:
            return

        # Generate samples
        self.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to(self.device) % 10
            init_noise = torch.randint(0, self.N, (16, 3, 32, 32)).to(self.device)

            images = self.d3pm.sample_with_image_sequence(init_noise, cond, stride=40)

            # Save image sequences as GIF
            gif = []
            for image in images:
                x_as_image = make_grid(image.float() / (self.N - 1), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            # Create contents directory if it doesn't exist
            os.makedirs("contents", exist_ok=True)

            gif_path = f"contents/cifar_sample_{self.global_step}.gif"
            png_path = f"contents/cifar_sample_{self.global_step}_last.png"

            gif[0].save(
                gif_path,
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(png_path)

            # Log images to wandb
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({
                    "generation_gif": wandb.Image(gif_path),
                    "final_image": wandb.Image(png_path),
                    "step": self.global_step
                })


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_epochs", type=int, default=4000)
    parser.add_argument("--N", type=int, default=8, help="Number of classes for discretized state")
    parser.add_argument("--hybrid_loss_coeff", type=float, default=0.0)
    parser.add_argument("--project", type=str, default="d3pm_cifar10", help="WandB project name")
    parser.add_argument("--name", type=str, default=None, help="WandB run name")

    # Add trainer specific arguments
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto",
                       help="Number of devices to use (int) or 'auto' for all available devices")
    parser.add_argument("--num_gpus", type=int, default=None,
                       help="Number of GPUs to use (overrides --devices)")
    parser.add_argument("--strategy", type=str, default="auto",
                       help="Training strategy: 'ddp', 'deepspeed', etc.")
    parser.add_argument("--precision", type=str, default="32-true")

    args = parser.parse_args()

    # Convert devices to the right format if num_gpus is specified
    devices = args.devices
    if args.num_gpus is not None:
        devices = args.num_gpus
        # Automatically set strategy to ddp when using multiple GPUs
        if args.num_gpus > 1 and args.strategy is None:
            args.strategy = "ddp"

    # Initialize wandb run first
    run_name = args.name if args.name else f"d3pm_cifar10_N{args.N}_bs{args.batch_size}_lr{args.lr}"

    # Create config dictionary
    config = {
        "N": args.N,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "hybrid_loss_coeff": args.hybrid_loss_coeff,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "devices": devices,
        "strategy": args.strategy,
        "precision": args.precision
    }

    # Then create the wandb logger using the initialized run
    wandb_logger = WandbLogger(project=args.project, name=run_name, log_model="all", config=config)

    # Initialize model and data module
    model = CIFAR10D3PMLightning(
        N=args.N,
        hybrid_loss_coeff=args.hybrid_loss_coeff,
        lr=args.lr
    )

    dm = CIFAR10DataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="cifar10-{epoch}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

    # Initialize the trainer with WandB logger
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        precision=args.precision,
        strategy=args.strategy if args.strategy else None,
        devices=devices,
        accelerator=args.accelerator
    )

    # Train the model
    trainer.fit(model, dm)

    # Close wandb run
    wandb.finish()
