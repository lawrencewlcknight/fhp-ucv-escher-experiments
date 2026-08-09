"""Lightweight, playable average-policy snapshots for the VR-Deep solvers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from open_spiel.python import policy

from .solver import MLP


SNAPSHOT_TYPE = "vr_deep_cfr_policy_snapshot"
SNAPSHOT_VERSION = 1


def snapshot_filename(algorithm_id: str, seed: int) -> str:
    return f"{algorithm_id}_seed_{int(seed)}_final_policy_snapshot.pt"


def save_policy_snapshot(
    solver,
    path: str | Path,
    *,
    algorithm_id: str,
    algorithm_label: str,
    seed: int,
    config: Dict[str, Any],
) -> Path:
    """Save only the fitted final average-policy and provenance metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in solver.ave_policy_trainer.model.state_dict().items()
    }
    torch.save(
        {
            "version": SNAPSHOT_VERSION,
            "type": SNAPSHOT_TYPE,
            "algorithm_id": str(algorithm_id),
            "algorithm_label": str(algorithm_label),
            "game": str(config.get("game_name", "FHP")),
            "seed": int(seed),
            "nodes_touched": int(solver.nodes_touched),
            "iteration": int(solver.num_iteration),
            "policy_state_dict": state_dict,
            "policy_network_layers": list(solver.network_layers),
            "input_size": int(solver.infostate_size),
            "num_actions": int(solver.action_size),
            "authors_parameterisation": dict(config),
        },
        path,
    )
    return path


class LoadedVRPolicy(policy.Policy):
    """OpenSpiel policy backed by a saved VR-Deep average-policy network."""

    def __init__(self, game, snapshot_path: str | Path):
        super().__init__(game, list(range(game.num_players())))
        self.path = Path(snapshot_path)
        snapshot = torch.load(self.path, map_location="cpu", weights_only=False)
        if snapshot.get("type") != SNAPSHOT_TYPE:
            raise ValueError(f"Not a VR-Deep policy snapshot: {self.path}")
        self.metadata = {
            key: value for key, value in snapshot.items() if key != "policy_state_dict"
        }
        self.model = MLP(
            int(snapshot["input_size"]),
            [int(value) for value in snapshot["policy_network_layers"]],
            int(snapshot["num_actions"]),
        )
        self.model.load_state_dict(snapshot["policy_state_dict"], strict=True)
        self.model.eval()

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal_actions = list(state.legal_actions(player))
        if not legal_actions:
            return {}
        info_state = torch.as_tensor(
            state.information_state_tensor(player), dtype=torch.float32
        )
        legal_mask = torch.as_tensor(
            state.legal_actions_mask(player), dtype=torch.float32
        )
        with torch.no_grad():
            logits = self.model(info_state)
            legal_logits = torch.where(
                legal_mask == 1,
                logits,
                torch.full_like(logits, -1e21),
            )
            probabilities = torch.softmax(legal_logits, dim=-1).cpu().numpy()
        selected = {action: float(probabilities[action]) for action in legal_actions}
        total = float(sum(selected.values()))
        if total <= 0 or not np.isfinite(total):
            raise ValueError(f"Invalid policy probabilities in {self.path}")
        return {action: value / total for action, value in selected.items()}


__all__ = [
    "LoadedVRPolicy",
    "SNAPSHOT_TYPE",
    "save_policy_snapshot",
    "snapshot_filename",
]
