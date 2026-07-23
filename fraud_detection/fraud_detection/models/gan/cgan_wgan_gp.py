"""Conditional WGAN-GP for minority-class (fraud) tabular data augmentation.

Replaces the original unconditional, vanilla sigmoid GAN in
`AI_model_Py_Scripts/FraudDetectionUSingGAN.ipynb` (cells 2-3), where synthetic
rows are generated unconditionally and labels are only assigned afterward by
sampling the real fraud ratio (cell 9) -- that GAN never actually learns what
makes a row fraudulent, it just learns the marginal feature distribution.
Here the generator is conditioned directly on the label (noise concatenated
with the label's one-hot encoding), and the plain sigmoid discriminator is
replaced with a Wasserstein critic + gradient penalty, which avoids the
vanishing-gradient/mode-collapse failure modes that hit a vanilla GAN hardest
on exactly the class that matters here: the rare, minority fraud class.

Known limitation: categorical feature columns are one-hot encoded upstream
(`FittedPreprocessor`) and the generator's sigmoid output produces continuous
approximations of those one-hot vectors, not discrete draws -- a Gumbel-softmax
(or similar categorical-aware) head would fix this but is out of scope here.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # preprocessed features are MinMax/one-hot scaled to [0, 1]
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return self.net(torch.cat([noise, one_hot], dim=1))


class ConditionalCritic(nn.Module):
    """No BatchNorm: WGAN-GP's gradient penalty is computed per-sample on
    interpolated inputs, and BatchNorm's batch-level statistics corrupt that
    per-sample gradient (the standard WGAN-GP caveat)."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return self.net(torch.cat([x, one_hot], dim=1))


def _gradient_penalty(
    critic: ConditionalCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, device=device)
    interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    scores = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


@dataclass
class CWGANGPConfig:
    noise_dim: int = 32
    hidden_dim: int = 256
    num_classes: int = 2
    epochs: int = 50
    batch_size: int = 128
    n_critic: int = 5
    lambda_gp: float = 10.0
    lr: float = 1e-4
    device: str = "cpu"


class CWGANGPTrainer:
    """Trains a conditional WGAN-GP on already-preprocessed feature arrays
    (i.e. `FittedPreprocessor.transform(df)` output) and supports
    label-conditioned sample generation for augmentation.

    Batches are drawn via a class-balanced `WeightedRandomSampler` rather than
    plain shuffling: at ~3% fraud prevalence, plain shuffling would give the
    fraud-conditioned generator/critic path far too few gradient updates.
    """

    def __init__(self, input_dim: int, config: Optional[CWGANGPConfig] = None):
        self.config = config or CWGANGPConfig()
        self.device = torch.device(self.config.device)
        self.generator = ConditionalGenerator(
            self.config.noise_dim, self.config.num_classes, input_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic = ConditionalCritic(
            input_dim, self.config.num_classes, self.config.hidden_dim
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "CWGANGPTrainer":
        cfg = self.config
        X_t = torch.tensor(X, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.long)

        class_counts = np.bincount(y, minlength=cfg.num_classes).astype(np.float64)
        sample_weights = 1.0 / class_counts[y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(y), replacement=True)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=cfg.batch_size, sampler=sampler, drop_last=True
        )

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=cfg.lr, betas=(0.5, 0.9))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, betas=(0.5, 0.9))

        for epoch in range(cfg.epochs):
            epoch_loss_c, epoch_loss_g, n_batches = 0.0, 0.0, 0
            for real_x, real_y in loader:
                real_x, real_y = real_x.to(self.device), real_y.to(self.device)
                bs = real_x.size(0)

                for _ in range(cfg.n_critic):
                    noise = torch.randn(bs, cfg.noise_dim, device=self.device)
                    fake_x = self.generator(noise, real_y).detach()
                    loss_c = (
                        self.critic(fake_x, real_y).mean()
                        - self.critic(real_x, real_y).mean()
                        + cfg.lambda_gp * _gradient_penalty(self.critic, real_x, fake_x, real_y, self.device)
                    )
                    opt_c.zero_grad()
                    loss_c.backward()
                    opt_c.step()

                noise = torch.randn(bs, cfg.noise_dim, device=self.device)
                fake_x = self.generator(noise, real_y)
                loss_g = -self.critic(fake_x, real_y).mean()
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                epoch_loss_c += loss_c.item()
                epoch_loss_g += loss_g.item()
                n_batches += 1

            if verbose and (epoch % max(cfg.epochs // 10, 1) == 0 or epoch == cfg.epochs - 1):
                print(
                    f"[CWGAN-GP] epoch {epoch + 1}/{cfg.epochs} "
                    f"critic_loss={epoch_loss_c / n_batches:.4f} gen_loss={epoch_loss_g / n_batches:.4f}"
                )
        return self

    @torch.no_grad()
    def generate(self, label: int, n_samples: int) -> np.ndarray:
        self.generator.eval()
        noise = torch.randn(n_samples, self.config.noise_dim, device=self.device)
        labels = torch.full((n_samples,), label, dtype=torch.long, device=self.device)
        samples = self.generator(noise, labels).cpu().numpy()
        self.generator.train()
        return np.clip(samples, 0.0, 1.0)
