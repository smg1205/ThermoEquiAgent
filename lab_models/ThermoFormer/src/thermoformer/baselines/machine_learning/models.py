"""Compact, audited neural architectures used by the VLE baseline adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class DescriptorANN(nn.Module):
    """Author topology: 21 descriptors + P + x, then 64-64-64-output."""

    INPUT_DIM = 23

    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.INPUT_DIM:
            raise ValueError("Descriptor ANN requires exactly 21 descriptors plus P and x")
        return self.network(values)


class SMILESRNN(nn.Module):
    """Direction-specific character-one-hot RNN matching the published layer widths."""

    def __init__(self, input_dim: int, output_dim: int = 2) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("SMILES-RNN input dimension must be positive")
        self.input_dim = input_dim
        self.rnn1 = nn.RNN(input_dim, 32, batch_first=True)
        self.rnn2 = nn.RNN(32, 16, batch_first=True)
        self.head = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, output_dim))

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(1)
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError("SMILES-RNN input must have shape [batch, sequence, features]")
        hidden, _ = self.rnn1(values)
        hidden, _ = self.rnn2(hidden)
        return self.head(hidden[:, -1])


class Set2SetPool(nn.Module):
    """Set2Set graph pooling with a recurrent query, as used by UALF-GNN."""

    def __init__(self, feature_dim: int, processing_steps: int = 3) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.processing_steps = processing_steps
        self.lstm = nn.LSTM(feature_dim * 2, feature_dim, batch_first=True)

    def forward(self, nodes: Tensor, mask: Tensor) -> Tensor:
        batch = nodes.shape[0]
        query_star = torch.zeros(batch, self.feature_dim * 2, dtype=nodes.dtype, device=nodes.device)
        hidden: tuple[Tensor, Tensor] | None = None
        for _ in range(self.processing_steps):
            query, hidden = self.lstm(query_star.unsqueeze(1), hidden)
            query = query[:, 0]
            logits = (nodes * query.unsqueeze(1)).sum(-1)
            logits = logits.masked_fill(~mask.bool(), -torch.inf)
            attention = torch.softmax(logits, dim=1)
            readout = (attention.unsqueeze(-1) * nodes).sum(1)
            query_star = torch.cat([query, readout], dim=-1)
        return query_star


class GraphConvGRUEncoder(nn.Module):
    """UALF molecular encoder with three GraphConv-ReLU-GRU updates."""

    def __init__(self, atom_dim: int = 16, hidden_dim: int = 32, steps: int = 3) -> None:
        super().__init__()
        self.input_projection = nn.Linear(atom_dim, hidden_dim)
        self.message_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(steps))
        self.gru_layers = nn.ModuleList(nn.GRUCell(hidden_dim, hidden_dim) for _ in range(steps))
        self.pool = Set2SetPool(hidden_dim * 2, processing_steps=3)

    def forward(self, atoms: Tensor, adjacency: Tensor, atom_mask: Tensor) -> Tensor:
        hidden = torch.relu(self.input_projection(atoms))
        initial = hidden
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        for message_layer, gru in zip(self.message_layers, self.gru_layers):
            message = torch.relu(message_layer(torch.bmm(adjacency, hidden) / degree))
            hidden = gru(message.reshape(-1, message.shape[-1]), hidden.reshape(-1, hidden.shape[-1]))
            hidden = hidden.reshape_as(message)
        joined = torch.cat([initial, hidden], dim=-1)
        return self.pool(joined, atom_mask)


class UALFGNN(nn.Module):
    """Paper-reimplemented binary heteroscedastic direct-VLE predictor."""

    def __init__(self, atom_dim: int = 16, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = GraphConvGRUEncoder(atom_dim, hidden_dim, steps=3)
        mixture_dim = hidden_dim * 8 + 2
        self.regressor = nn.Sequential(
            nn.Linear(mixture_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def forward(
        self,
        atoms: Tensor,
        adjacency: Tensor,
        atom_mask: Tensor,
        pressure_and_x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if atoms.shape[1] != 2:
            raise ValueError("UALF-GNN is a native binary model")
        vectors = [
            self.encoder(atoms[:, index], adjacency[:, index], atom_mask[:, index])
            for index in range(2)
        ]
        values = self.regressor(torch.cat([*vectors, pressure_and_x], dim=-1))
        mean, log_variance = values.chunk(2, dim=-1)
        return mean, log_variance.clamp(-12.0, 12.0)

    @staticmethod
    def heteroscedastic_loss(mean: Tensor, log_variance: Tensor, target: Tensor) -> Tensor:
        return 0.5 * (torch.exp(-log_variance) * (target - mean).square() + log_variance).mean()

    def mc_predict(
        self,
        atoms: Tensor,
        adjacency: Tensor,
        atom_mask: Tensor,
        pressure_and_x: Tensor,
        samples: int = 32,
    ) -> tuple[Tensor, Tensor]:
        if samples < 2:
            raise ValueError("MC dropout requires at least two samples")
        previous = self.training
        self.train()
        with torch.no_grad():
            outputs = [
                self(atoms, adjacency, atom_mask, pressure_and_x)
                for _ in range(samples)
            ]
        self.train(previous)
        means = torch.stack([value[0] for value in outputs])
        aleatoric = torch.stack([torch.exp(value[1]) for value in outputs]).mean(0)
        predictive_variance = means.var(0, unbiased=True) + aleatoric
        return means.mean(0), predictive_variance


class MolecularInteractionNetwork(nn.Module):
    """Shared PyTorch port of the SolvGNN/GDI-GNN molecular interaction block."""

    def __init__(self, molecule_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_projection = nn.Linear(molecule_dim + 1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.edge_network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        molecule_vectors: Tensor,
        x: Tensor,
        mask: Tensor,
        hydrogen_bond_edges: Tensor,
    ) -> Tensor:
        nodes = torch.relu(self.node_projection(torch.cat([molecule_vectors, x.unsqueeze(-1)], dim=-1)))
        expected = (*x.shape, x.shape[1])
        if hydrogen_bond_edges.shape != expected:
            raise ValueError(f"Hydrogen-bond edge tensor must have shape {expected}")
        matrices = self.edge_network(hydrogen_bond_edges.unsqueeze(-1)).reshape(
            *hydrogen_bond_edges.shape, self.hidden_dim, self.hidden_dim
        )
        messages = torch.einsum("bijhk,bjk->bih", matrices, nodes)
        messages = messages * mask.unsqueeze(-1)
        updated = self.gru(messages.reshape(-1, messages.shape[-1]), nodes.reshape(-1, nodes.shape[-1]))
        return updated.reshape_as(nodes) * mask.unsqueeze(-1)


class MolecularGraphEncoder(nn.Module):
    """Two-layer molecular GraphConv encoder used by the official graph baselines."""

    def __init__(self, atom_dim: int = 74, hidden_dim: int = 32) -> None:
        super().__init__()
        self.first = nn.Linear(atom_dim, hidden_dim)
        self.second = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atoms: Tensor, adjacency: Tensor, atom_mask: Tensor) -> Tensor:
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        hidden = torch.relu(self.first(torch.bmm(adjacency, atoms) / degree))
        hidden = torch.relu(self.second(torch.bmm(adjacency, hidden) / degree))
        weighted = hidden * atom_mask.unsqueeze(-1)
        return weighted.sum(1) / atom_mask.sum(1, keepdim=True).clamp_min(1.0)


class SolvGNN(nn.Module):
    def __init__(self, atom_dim: int = 74, molecule_dim: int = 32) -> None:
        super().__init__()
        self.molecular_encoder = MolecularGraphEncoder(atom_dim, molecule_dim)
        self.interaction = MolecularInteractionNetwork(molecule_dim)
        self.output = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_vectors(
        self,
        molecule_vectors: Tensor,
        x: Tensor,
        mask: Tensor,
        hydrogen_bond_edges: Tensor,
    ) -> Tensor:
        nodes = self.interaction(molecule_vectors, x, mask, hydrogen_bond_edges)
        return self.output(nodes).squeeze(-1) * mask

    def forward(
        self,
        atoms: Tensor,
        adjacency: Tensor,
        atom_mask: Tensor,
        x: Tensor,
        mixture_mask: Tensor,
        hydrogen_bond_edges: Tensor,
    ) -> Tensor:
        vectors = torch.stack(
            [
                self.molecular_encoder(
                    atoms[:, component], adjacency[:, component], atom_mask[:, component]
                )
                for component in range(atoms.shape[1])
            ],
            dim=1,
        )
        return self.forward_vectors(vectors, x, mixture_mask, hydrogen_bond_edges)


class GDIGNN(SolvGNN):
    """Selected GDI-GNN_xMLP: composition enters after molecular message passing."""

    def __init__(self, atom_dim: int = 74, molecule_dim: int = 32) -> None:
        super().__init__(atom_dim, molecule_dim)
        self.output = nn.Sequential(
            nn.Linear(65, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
        )

    def forward_vectors(
        self,
        molecule_vectors: Tensor,
        x: Tensor,
        mask: Tensor,
        hydrogen_bond_edges: Tensor,
    ) -> Tensor:
        nodes = self.interaction(
            molecule_vectors,
            torch.zeros_like(x),
            mask,
            hydrogen_bond_edges,
        )
        return self.output(torch.cat([nodes, x.unsqueeze(-1)], dim=-1)).squeeze(-1) * mask

    @staticmethod
    def gibbs_duhem_residual(log_gamma: Tensor, x: Tensor) -> Tensor:
        if not x.requires_grad:
            raise ValueError("Composition must require gradients for the GDI loss")
        derivatives = []
        for index in range(log_gamma.shape[1]):
            gradient = torch.autograd.grad(log_gamma[:, index].sum(), x, create_graph=True, retain_graph=True)[0]
            derivatives.append(gradient)
        jacobian = torch.stack(derivatives, dim=1)
        return (x.unsqueeze(1) * jacobian).sum(1)


class GEGNN(nn.Module):
    """Symmetric binary excess-Gibbs model with activity coefficients by autograd."""

    def __init__(self, atom_dim: int = 74, molecule_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.molecular_encoder = MolecularGraphEncoder(atom_dim, molecule_dim)
        self.interaction = MolecularInteractionNetwork(molecule_dim, hidden_dim)
        self.component_transform = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Softplus(),
        )
        self.energy = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Softplus(), nn.Linear(hidden_dim, 1))

    def ge_over_rt(
        self,
        molecule_vectors: Tensor,
        x: Tensor,
        hydrogen_bond_edges: Tensor | None = None,
    ) -> Tensor:
        if molecule_vectors.shape[1] != 2:
            raise ValueError("Native GE-GNN is binary")
        if hydrogen_bond_edges is None:
            hydrogen_bond_edges = torch.zeros(
                x.shape[0], 2, 2, dtype=x.dtype, device=x.device
            )
        mask = torch.ones_like(x)
        nodes = self.interaction(
            molecule_vectors,
            torch.zeros_like(x),
            mask,
            hydrogen_bond_edges,
        )
        symmetric = self.component_transform(torch.cat([nodes, x.unsqueeze(-1)], dim=-1)).mean(1)
        return self.energy(symmetric).squeeze(-1)

    def forward(
        self,
        molecule_vectors: Tensor,
        x: Tensor,
        hydrogen_bond_edges: Tensor | None = None,
    ) -> Tensor:
        x1 = x[:, 0] if x.requires_grad else x[:, 0].detach().clone().requires_grad_(True)
        composition = torch.stack([x1, 1.0 - x1], dim=-1)
        energy = self.ge_over_rt(molecule_vectors, composition, hydrogen_bond_edges)
        derivative = torch.autograd.grad(
            energy.sum(), x1, create_graph=self.training
        )[0]
        return torch.stack(
            [
                energy + (1.0 - x1) * derivative,
                energy - x1 * derivative,
            ],
            dim=-1,
        )

    def forward_graphs(
        self,
        atoms: Tensor,
        adjacency: Tensor,
        atom_mask: Tensor,
        x: Tensor,
        hydrogen_bond_edges: Tensor,
    ) -> Tensor:
        if atoms.shape[1] != 2:
            raise ValueError("Native GE-GNN is binary")
        vectors = torch.stack(
            [
                self.molecular_encoder(
                    atoms[:, component], adjacency[:, component], atom_mask[:, component]
                )
                for component in range(2)
            ],
            dim=1,
        )
        return self(vectors, x, hydrogen_bond_edges)


def muggianu_binary_weights(x: Tensor) -> Tensor:
    """Return normalized pair weights used by HANNA's geometric projection."""
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("Muggianu projection requires [batch, components] compositions")
    pairs = []
    for first in range(x.shape[1]):
        for second in range(first + 1, x.shape[1]):
            denominator = (x[:, first] + x[:, second]).clamp_min(1e-12)
            pairs.append((x[:, first] * x[:, second] / denominator).unsqueeze(-1))
    weights = torch.cat(pairs, dim=-1)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True)
class ExternalModelAssets:
    checkpoint: str
    overlap_inventory: str

    @property
    def complete(self) -> bool:
        from pathlib import Path

        return bool(self.checkpoint and self.overlap_inventory) and all(
            Path(path).is_file() for path in (self.checkpoint, self.overlap_inventory)
        )
