"""Reloadable average-policy checkpoints for FHP UCV-ESCHER."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import pickle
from typing import Mapping, Optional

import numpy as np
from open_spiel.python import policy

from .game import FHP_GAME_PARAMETERS, serialisable_game_definition


CHECKPOINT_TYPE = "fhp_ucv_escher_policy_checkpoint"
CHECKPOINT_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_policy_checkpoint(
    solver,
    path: str | Path,
    *,
    seed: int,
    config: Mapping[str, object],
    checkpoint_row: Mapping[str, object],
) -> Path:
    """Save a CPU-only policy checkpoint without advancing training RNGs."""
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in solver.ave_policy_trainer.model.state_dict().items()
    }
    payload = {
        "version": CHECKPOINT_VERSION,
        "type": CHECKPOINT_TYPE,
        "framework": "pytorch",
        "algorithm": "UCV-ESCHER",
        "algorithm_id": str(checkpoint_row.get("algorithm_id", "ucv_escher")),
        "execution_backend": str(
            checkpoint_row.get("execution_backend", "sequential")
        ),
        "experiment_id": checkpoint_row.get("experiment_id"),
        "experiment_name": checkpoint_row.get("experiment_name"),
        "game": serialisable_game_definition(),
        "seed": int(seed),
        "outer_iteration": int(solver.num_iteration),
        "episode": int(solver.episode),
        "nodes_touched": int(solver.nodes_touched),
        "checkpoint_kind": str(checkpoint_row.get("checkpoint_kind", "")),
        "checkpoint_target_nodes": checkpoint_row.get("checkpoint_target_nodes"),
        "checkpoint_target_seconds": checkpoint_row.get("checkpoint_target_seconds"),
        "training_elapsed_seconds": float(
            checkpoint_row.get("training_elapsed_seconds", 0.0)
        ),
        "wall_clock_seconds": float(checkpoint_row.get("wall_clock_seconds", 0.0)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_config": dict(config),
        "input_size": int(solver.infostate_size),
        "num_actions": int(solver.action_size),
        "policy_network_layers": list(
            getattr(solver, "average_policy_network_layers", solver.network_layers)
        ),
        "policy_state_dict": state_dict,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_checkpoint_payload(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("type") != CHECKPOINT_TYPE:
        raise ValueError(f"Unsupported checkpoint type: {payload.get('type')!r}")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')!r}")
    parameters = payload.get("game", {}).get("parameters")
    if parameters != dict(FHP_GAME_PARAMETERS):
        raise ValueError("Checkpoint does not use the canonical FHP game definition")
    return payload


class LoadedFHPPolicy(policy.Policy):
    """OpenSpiel-compatible policy restored from a saved checkpoint."""

    def __init__(self, game, checkpoint_path: str | Path):
        import torch

        from vr_deep_cfr.solver import MLP

        super().__init__(game, list(range(game.num_players())))
        self.game = game
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = load_checkpoint_payload(checkpoint_path)
        self.model = MLP(
            int(self.checkpoint["input_size"]),
            [int(value) for value in self.checkpoint["policy_network_layers"]],
            int(self.checkpoint["num_actions"]),
        )
        self.model.load_state_dict(self.checkpoint["policy_state_dict"])
        self.model.eval()
        self._torch = torch

    def action_probabilities(self, state, player_id: Optional[int] = None):
        player = state.current_player() if player_id is None else int(player_id)
        legal_actions = state.legal_actions(player)
        if not legal_actions:
            return {}
        info_state = self._torch.as_tensor(
            state.information_state_tensor(player), dtype=self._torch.float32
        )
        legal_mask = self._torch.as_tensor(
            state.legal_actions_mask(player), dtype=self._torch.bool
        )
        with self._torch.no_grad():
            logits = self.model(info_state)
            masked_logits = self._torch.where(
                legal_mask,
                logits,
                self._torch.full_like(logits, -1e21),
            )
            probabilities = self._torch.softmax(masked_logits, dim=-1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Checkpoint policy produced non-finite probabilities")
        return {action: float(probabilities[action]) for action in legal_actions}
