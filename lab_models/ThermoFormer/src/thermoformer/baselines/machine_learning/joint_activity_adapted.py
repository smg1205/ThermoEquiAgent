"""Train-only residual calibration for frozen external activity models.

The adapter preserves the published molecular model and learns a small shared,
permutation-equivariant correction from the registered joint binary/ternary
training partition.  Validation labels select the checkpoint; test labels are
never used for fitting or selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ...data.loading import VLESample
from ...data.splitting import canonical_smiles


@dataclass(frozen=True)
class JointActivityConfig:
    hidden_dimension: int = 32
    dropout: float = 0.05
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 256
    epochs: int = 100
    patience: int = 15
    minimum_composition: float = 1.0e-5


class JointActivityResidualHead(nn.Module):
    """Shared component correction conditioned on mixture-level summaries."""

    def __init__(self, hidden_dimension: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, hidden_dimension),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, base: Tensor, temperature_k: Tensor, composition: Tensor) -> Tensor:
        if base.shape != composition.shape or base.ndim != 2:
            raise ValueError("base log-gamma and composition must have shape [batch, components]")
        count = base.shape[1]
        mean_base = base.mean(dim=1, keepdim=True).expand_as(base)
        mean_x = composition.mean(dim=1, keepdim=True).expand_as(composition)
        features = torch.stack(
            (
                base,
                composition,
                (base - mean_base).abs(),
                mean_base,
                mean_x,
                (1000.0 / temperature_k.reshape(-1, 1)).expand_as(base),
                torch.full_like(base, float(count) / 3.0),
            ),
            dim=-1,
        )
        return base + self.network(features).squeeze(-1)


def equilibrium_log_gamma(
    sample: VLESample,
    vapor_pressure: Mapping[str, object],
    minimum_composition: float,
) -> Tensor | None:
    x = np.asarray(sample.liquid_composition, dtype=np.float64)
    y = np.asarray(sample.vapor_composition, dtype=np.float64)
    if np.any(x <= minimum_composition) or np.any(y <= minimum_composition):
        return None
    log_psat = np.asarray(
        [
            vapor_pressure[canonical_smiles(smiles)].intercept
            + vapor_pressure[canonical_smiles(smiles)].inverse_temperature / sample.temperature_k
            for smiles in sample.smiles
        ],
        dtype=np.float64,
    )
    target = np.log(y) + np.log(sample.pressure_kpa) - np.log(x) - log_psat
    if not np.isfinite(target).all():
        return None
    return torch.as_tensor(target, dtype=torch.float32)


@dataclass
class JointActivityTrainingResult:
    head: JointActivityResidualHead
    history: tuple[dict[str, float], ...]
    best_epoch: int
    train_rows: int
    validation_rows: int


def train_joint_activity_head(
    train: Sequence[VLESample],
    validation: Sequence[VLESample],
    vapor_pressure: Mapping[str, object],
    base_predictor: object,
    config: JointActivityConfig,
    device: torch.device,
) -> JointActivityTrainingResult:
    """Fit the residual head to train-derived activity coefficients."""

    def materialize(rows: Sequence[VLESample]) -> dict[int, tuple[Tensor, Tensor, Tensor, Tensor]]:
        valid_samples: list[VLESample] = []
        targets: list[Tensor] = []
        for sample in rows:
            if not all(canonical_smiles(value) in vapor_pressure for value in sample.smiles):
                continue
            target = equilibrium_log_gamma(sample, vapor_pressure, config.minimum_composition)
            if target is not None:
                valid_samples.append(sample)
                targets.append(target)
        batch_predictor = getattr(base_predictor, "log_gamma_many", None)
        bases = (
            batch_predictor(valid_samples)
            if batch_predictor is not None
            else [
                base_predictor.log_gamma(
                    sample.smiles, sample.temperature_k, sample.liquid_composition
                ).detach().cpu()
                for sample in valid_samples
            ]
        )
        grouped: dict[int, list[tuple[Tensor, Tensor, Tensor, Tensor]]] = {2: [], 3: []}
        for sample, target, base in zip(valid_samples, targets, bases):
            grouped[sample.component_count].append(
                (
                    base,
                    torch.tensor(sample.liquid_composition, dtype=torch.float32),
                    torch.tensor([sample.temperature_k], dtype=torch.float32),
                    target,
                )
            )
        output = {}
        for count, values in grouped.items():
            if values:
                output[count] = tuple(torch.stack(items) for items in zip(*values))
        return output
    train_data = materialize(train)
    validation_data = materialize(validation)
    if not train_data or not validation_data:
        raise RuntimeError("Joint activity adaptation has no train/validation activity labels")
    head = JointActivityResidualHead(config.hidden_dimension, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(torch.initial_seed())
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    best_epoch, best_validation, stale = 0, float("inf"), 0
    history: list[dict[str, float]] = []
    training_started = time.perf_counter()
    train_rows = sum(len(values[0]) for values in train_data.values())
    validation_rows = sum(len(values[0]) for values in validation_data.values())
    print(
        f"[activity-adapted] starting {config.epochs} epoch(s); "
        f"train_rows={train_rows} validation_rows={validation_rows}",
        flush=True,
    )

    def loss_on(grouped: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]], training: bool) -> Tensor:
        losses = []
        for base, x, temperature, target in grouped.values():
            order = torch.randperm(len(base), generator=generator) if training else torch.arange(len(base))
            for start in range(0, len(order), config.batch_size):
                idx = order[start : start + config.batch_size]
                prediction = head(
                    base[idx].to(device), temperature[idx].to(device), x[idx].to(device)
                )
                losses.append(torch.mean((prediction - target[idx].to(device)) ** 2))
        return torch.stack(losses).mean()

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss_on(train_data, True)
        train_loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            validation_loss = float(loss_on(validation_data, False).cpu())
        history.append(
            {"epoch": float(epoch), "train_loss": float(train_loss.detach().cpu()), "validation_loss": validation_loss}
        )
        improved = validation_loss < best_validation - 1.0e-8
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
        now = time.perf_counter()
        print(
            f"[activity-adapted] [epoch {epoch}/{config.epochs}] "
            f"train_loss={float(train_loss.detach().cpu()):.6f} "
            f"val_loss={validation_loss:.6f} best={best_validation:.6f} "
            f"improved={'yes' if improved else 'no'} stale={stale}/{config.patience} "
            f"epoch_time={now - epoch_started:.1f}s "
            f"total_time={now - training_started:.1f}s",
            flush=True,
        )
        if stale >= config.patience:
            print(
                f"[activity-adapted] early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch} best_validation={best_validation:.6f}",
                flush=True,
            )
            break
    head.load_state_dict(best_state)
    head.eval()
    return JointActivityTrainingResult(
        head=head,
        history=tuple(history),
        best_epoch=best_epoch,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )


class JointAdaptedActivityPredictor:
    """Apply a selected residual head to a frozen official predictor."""

    def __init__(self, base_predictor: object, head: JointActivityResidualHead, device: torch.device) -> None:
        self.base_predictor = base_predictor
        self.head = head.to(device).eval()
        self.device = device

    def log_gamma(self, smiles: Sequence[str], temperature_k: float, composition: Sequence[float]) -> Tensor:
        base = self.base_predictor.log_gamma(smiles, temperature_k, composition).to(self.device)
        x = torch.as_tensor(composition, dtype=torch.float32, device=self.device).reshape(1, -1)
        temperature = torch.tensor([[temperature_k]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.head(base.reshape(1, -1), temperature, x)[0]
        return prediction.detach().cpu()
