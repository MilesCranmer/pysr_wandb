"""CLI for PySR-W&B."""
from .pysr_wandb import init_wandb, runall

if __name__ == "__main__":
    wandb = init_wandb()
    runall(wandb)
