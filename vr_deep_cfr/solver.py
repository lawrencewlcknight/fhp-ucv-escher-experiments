import math
import random
import time
from typing import Any, Union

import numpy as np
import pyspiel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats

from fhp_escher.game import load_fhp_game
from .logger import Logger

SpielState = Union[pyspiel.State, Any]


def set_seed(seed):
    """Match the upstream implementation's NumPy/Python/PyTorch seeding."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True


class DeepCumuAdv:
    def __init__(
        self,
        game_name,
        num_episodes=400,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        learning_rate=1e-4,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        use_regret_matching_argmax=True,
        epsilon=0.6,
        logger=None,
        fit_advantage=True,
        use_baseline=False,
        baseline_buffer_size=100_0000,
        baseline_batch_size=-1,
        baseline_network_train_steps=1,
        gamma=0,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        self.game_name = game_name
        self.play_against_random = play_against_random
        self.logger = logger or Logger(writer_strings=[])
        self.game = self.load_game()
        self.num_players = self.game.num_players()
        self.infostate_size = self.game.information_state_tensor_size()
        self.action_size = self.game.num_distinct_actions()
        self.advantage_buffer_size = advantage_buffer_size
        self.ave_policy_buffer_size = ave_policy_buffer_size
        self.num_episodes = num_episodes
        self.num_traversals = num_traversals
        self.num_iterations = self.num_episodes // (self.num_traversals * 2)
        self.advantage_network_train_steps = advantage_network_train_steps
        self.ave_policy_network_train_steps = ave_policy_network_train_steps
        self.ave_policy_batch_size = ave_policy_batch_size
        self.advantage_batch_size = advantage_batch_size
        self.reinitialize_advantage_networks = reinitialize_advantage_networks
        self.learning_rate = learning_rate
        self.device = device
        self.gamma = gamma
        self.evaluation_frequency = evaluation_frequency
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.max_utility = max(self.game.max_utility(), abs(self.game.min_utility()))
        self.network_layers = [num_hiddens for _ in range(num_layers)]
        self.epsilon = epsilon
        self.num_random_games = num_random_games
        self.fit_advantage = fit_advantage
        self.use_baseline = use_baseline
        self.baseline_buffer_size = baseline_buffer_size
        self.baseline_network_train_steps = baseline_network_train_steps
        self.baseline_batch_size = baseline_batch_size
        self.num_iteration = 0
        self.nodes_touched = 0
        self.episode = 0
        self.target_nodes_touched = None
        self.max_num_iterations = None
        self.preserve_evaluation_rng = True
        self.evaluate_initial_policy = False
        self.early_evaluation_node_thresholds = ()
        self.checkpoint_rows = []
        self._solve_start_time = None
        self._next_early_evaluation_index = 0
        self._last_average_policy_loss = np.nan
        self.max_wall_clock_seconds = None
        self.stop_reason = "iteration_cap"
        self._last_iteration_seconds = None
        set_seed(seed)
        self.init_ave_policy_trainer()
        self.init_regret_trainers()

        if self.use_baseline:
            self.init_q_value_trainer()

    def init_ave_policy_trainer(self):
        self.ave_policy_trainer = AvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.ave_policy_network_train_steps,
            self.logger,
            self.device,
            self.gamma,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [
            RegretTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.device,
            )
            for _ in range(self.num_players)
        ]

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_tensor = np.append(
            root_state.information_state_tensor(0),
            root_state.information_state_tensor(1),
        )
        history_size = len(history_tensor)

        self.q_value_trainer = QValueTrainer(
            history_size,
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.baseline_buffer_size,
            self.baseline_batch_size,
            self.baseline_network_train_steps,
            self.logger,
            self.regret_trainers,
            self.device,
        )

    def solve(self, post_checkpoint_callback=None):
        """Train until the configured iteration cap or matched node budget.

        Optional initial and node-threshold checkpoints are evaluation-only:
        they do not add training nodes and their RNG use is isolated from the
        subsequent traversal stream. The optional callback is called after an
        already scheduled policy fit and diagnostic checkpoint. It receives the
        solver and checkpoint row and does not require another fit or
        evaluation.
        """
        self._post_checkpoint_callback = post_checkpoint_callback
        self._solve_start_time = time.perf_counter()
        self._prepare_early_evaluation_schedule()
        if self.evaluate_initial_policy:
            # The untrained average-policy head is zero-initialised, so this is
            # the exact uniform-over-legal-actions policy. No replay fit is
            # possible or necessary before the first training sample exists.
            self.evaluate(checkpoint_kind="initial_untrained_policy")
        iteration_cap = self.num_iterations
        if self.max_num_iterations is not None:
            iteration_cap = min(iteration_cap, int(self.max_num_iterations))
        while self.num_iteration < iteration_cap:
            if self._should_stop_for_wall_clock_budget():
                self.stop_reason = "wall_clock_budget"
                break
            iteration_start = time.perf_counter()
            self.iteration()
            self._last_iteration_seconds = time.perf_counter() - iteration_start
            if (
                self.target_nodes_touched is not None
                and self.nodes_touched >= int(self.target_nodes_touched)
            ):
                self.stop_reason = "node_budget"
                break
            if self._wall_clock_seconds() >= self._wall_clock_limit():
                self.stop_reason = "wall_clock_budget"
                break
        if not self.checkpoint_rows or (
            self.checkpoint_rows[-1]["nodes_touched"] != self.nodes_touched
        ):
            self._run_checkpoint(checkpoint_kind="final_node_budget")
        return list(self.checkpoint_rows)

    def _wall_clock_seconds(self):
        if self._solve_start_time is None:
            return 0.0
        return time.perf_counter() - self._solve_start_time

    def _wall_clock_limit(self):
        if self.max_wall_clock_seconds is None:
            return float("inf")
        return float(self.max_wall_clock_seconds)

    def _should_stop_for_wall_clock_budget(self):
        """Avoid starting an iteration unlikely to finish before the cap."""
        elapsed = self._wall_clock_seconds()
        limit = self._wall_clock_limit()
        if elapsed >= limit:
            return True
        if self._last_iteration_seconds is None:
            return False
        return elapsed + float(self._last_iteration_seconds) >= limit

    def iteration(self):
        self.num_iteration += 1
        for player in range(self.num_players):
            self.collect_training_data(player)
            self.train_regret(player)
            if self.use_baseline:
                self.train_baseline(player)
        if self.num_iteration % self.evaluation_frequency == 0:
            self._run_checkpoint(checkpoint_kind="outer_iteration")

    def _prepare_early_evaluation_schedule(self):
        thresholds = tuple(
            sorted({int(value) for value in self.early_evaluation_node_thresholds})
        )
        if any(value <= 0 for value in thresholds):
            raise ValueError("Early evaluation node thresholds must be positive.")
        self.early_evaluation_node_thresholds = thresholds
        self._next_early_evaluation_index = 0

    def _maybe_run_early_node_checkpoint(self):
        """Evaluate once a configured training-node threshold is crossed."""
        while self._next_early_evaluation_index < len(
            self.early_evaluation_node_thresholds
        ):
            threshold = self.early_evaluation_node_thresholds[
                self._next_early_evaluation_index
            ]
            if self.nodes_touched < threshold:
                return
            self._next_early_evaluation_index += 1
            if len(self.ave_policy_trainer.buffer) == 0:
                # This can only occur for an unusually tiny threshold. Defer
                # until a complete trajectory has supplied average-policy data.
                self._next_early_evaluation_index -= 1
                return
            self._run_checkpoint(
                checkpoint_kind="early_node_threshold",
                checkpoint_target_nodes=threshold,
            )

    def _capture_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch"])
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _run_checkpoint(
        self,
        *,
        checkpoint_kind="outer_iteration",
        checkpoint_target_nodes=None,
    ):
        """Fit/evaluate the average policy without perturbing training RNG."""
        rng_state = self._capture_rng_state() if self.preserve_evaluation_rng else None
        try:
            self.train_average_policy()
            self.evaluate(
                checkpoint_kind=checkpoint_kind,
                checkpoint_target_nodes=checkpoint_target_nodes,
            )
            callback = getattr(self, "_post_checkpoint_callback", None)
            if callback is not None:
                callback(self, dict(self.checkpoint_rows[-1]))
        finally:
            if rng_state is not None:
                self._restore_rng_state(rng_state)

    def collect_training_data(self, player):
        self.regret_trainers[player].reset_buffer()
        for _ in range(self.num_traversals):
            self.episode += 1
            root_state = self.skip_chance_state(self.game.new_initial_state())
            self.dfs(root_state, player)
            self._maybe_run_early_node_checkpoint()

    def train_regret(self, player):
        if self.reinitialize_advantage_networks:
            self.regret_trainers[player].reset()
        elif hasattr(self.regret_trainers[player], "reset_immediate_regret"):
            # VR-DeepPDCFR+ reinitialises only the immediate-regret network in
            # the paper configuration. Upstream coupled this to cumulative
            # network reinitialisation, which made the setting ineffective.
            self.regret_trainers[player].reset_immediate_regret()
        regret_loss = self.regret_trainers[player].train_model(self.num_iteration)
        self.logger.record("regret_loss_{}".format(player), regret_loss)

    def train_baseline(self, player):
        baseline_loss = self.q_value_trainer.train_model(self.num_iteration)
        if baseline_loss is not None:
            self.logger.record("baseline_loss_{}".format(player), baseline_loss)

    def train_average_policy(self):
        self.ave_policy_trainer.reset()
        ave_policy_loss = self.ave_policy_trainer.train_model(self.num_iteration)
        self._last_average_policy_loss = ave_policy_loss
        self.logger.record("average_policy_loss", ave_policy_loss)
        self.logger.info("average policy loss: {}".format(ave_policy_loss))

    def evaluate(
        self,
        *,
        checkpoint_kind="outer_iteration",
        checkpoint_target_nodes=None,
    ):
        self.logger.record("nodes_touched", self.nodes_touched)
        self.logger.record("iteration", self.num_iteration)
        self.logger.record("episode", self.episode)
        self.logger.record("checkpoint_kind", checkpoint_kind)
        self.logger.record("checkpoint_target_nodes", checkpoint_target_nodes)
        # Exact tabular exploitability requires enumerating FHP's enormous game
        # tree. Checkpoints therefore report training diagnostics and are kept
        # reloadable for sampled head-to-head evaluation instead.
        self.logger.record("exp", np.nan)
        self.logger.record("average_policy_value", np.nan)
        self.logger.record("evaluation_status", "deferred_to_head_to_head")
        wall_clock = self._wall_clock_seconds()
        self.logger.record("wall_clock_seconds", wall_clock)
        row = self.logger.dump(step=self.episode)
        self.checkpoint_rows.append(dict(row))

    def dfs(
        self,
        s,
        traverser,
        my_reach=1.0,
        opp_reach=1.0,
        opp_sample_reach=1.0,
        sample_reach=1.0,
    ):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s, self.num_iteration)
        num_actions = s.num_distinct_actions()
        uniform_policy = np.array(s.legal_actions_mask()) / len(s.legal_actions())
        sample_policy = (
            uniform_policy * self.epsilon + policy * (1 - self.epsilon)
            if player == traverser
            else policy
        )
        sample_policy /= sample_policy.sum()
        action = np.random.choice(range(num_actions), p=sample_policy)
        sample_prob, prob = (
            sample_policy[action].item(),
            policy[action].item(),
        )
        ns = self.skip_chance_state(s.child(action))
        if player == 1 - traverser:
            self.ave_policy_trainer.add_data(
                self.get_infostate_tensor(s),
                policy,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            if self.use_baseline:
                q_values = self.q_value_trainer.get_baseline(s, traverser)
            else:
                q_values = np.zeros_like(policy)
            action_value = self.dfs(
                ns,
                traverser,
                my_reach,
                opp_reach * prob,
                opp_sample_reach * sample_prob,
                sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)
        else:
            if self.use_baseline:
                q_values = self.q_value_trainer.get_baseline(s, traverser)
            else:
                q_values = np.zeros_like(policy)
            action_value = self.dfs(
                ns,
                traverser,
                my_reach * prob,
                opp_reach,
                opp_sample_reach,
                sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)

            cf_regrets = np.zeros_like(policy)
            im_weight = 1 if self.fit_advantage else opp_reach / sample_reach
            cf_regrets[legal_actions] = -value * im_weight
            cf_regrets[legal_actions] += q_values[legal_actions] * im_weight
            self.regret_trainers[player].add_data(
                self.get_infostate_tensor(s),
                cf_regrets,
                s.legal_actions_mask(),
                self.num_iteration,
            )

        if self.use_baseline:
            self.q_value_trainer.add_data(
                self.get_history_tensor(s),
                action,
                self.get_history_tensor(ns),
                self.get_infostate_tensor(ns) if not ns.is_terminal() else None,
                ns.legal_actions_mask() if not ns.is_terminal() else None,
                ns.current_player(),
                int(ns.is_terminal()),
                ns.returns()[0] / self.max_utility,
            )
        return value

    def get_infostate_tensor(self, s):
        return s.information_state_tensor()

    def get_history_tensor(self, s):
        return np.append(
            s.information_state_tensor(0), s.information_state_tensor(1)
        )
    def load_game(self):
        if self.game_name != "FHP":
            raise ValueError(
                f"This repository supports the FHP game definition only, got "
                f"{self.game_name!r}."
            )
        self.poker_game = True
        return load_fhp_game()

    def skip_chance_state(self, s):
        while s.current_player() == -1:
            self.nodes_touched += 1
            actions, probs = zip(*s.chance_outcomes())
            aid = np.random.choice(range(len(actions)), p=probs)
            s.apply_action(actions[aid])
        return s

class Trainer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.logger = logger
        self.device = device

        self.model = self.init_model()
        self.buffer = ReservoirBuffer(
            self.buffer_size, self.input_size, self.output_size, device=self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.softmax_fn = nn.Softmax(dim=-1)

    def init_model(self):
        model = MLP(self.input_size, self.network_layers, self.output_size).to(
            self.device
        )
        return model
    def reset(self):
        self.model.reset_parameters()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def reset_buffer(self):
        self.buffer.reset()

    def add_data(self, infostate, q_value, q_value_mask, iteration):
        self.buffer.add(infostate, q_value, q_value_mask, iteration)

    def get_infostate_tensor(self, s):
        return s.information_state_tensor()

    def get_history_tensor(self, s):
        return np.append(
            s.information_state_tensor(0), s.information_state_tensor(1)
        )

class RegretTrainer(Trainer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        use_regret_matching_argmax: bool,
        device: str = "cpu",
    ):
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            device,
        )
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())

    def forward(self, model, x, mask):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            with torch.no_grad():
                legal_regrets = self.predict(model, x, mask)
                return legal_regrets.cpu().numpy()

    def predict(self, model, x, mask):
        legal_regrets = model(x) * mask
        return legal_regrets

    def train_model(self, T):
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] regret loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def compute_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        with torch.no_grad():
            target_outputs = self.predict(
                self.target_model, infostates, legal_actions_mask
            )
        regrets = target_outputs + cf_regrets
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, regrets)
        return loss

    def get_policy(self, s: SpielState, T: int) -> np.ndarray:
        regrets = self.get_regrets(s)
        legal_actions = s.legal_actions()
        return self.regret_matching(regrets, legal_actions)

    def get_regrets(self, s: SpielState) -> np.ndarray:
        regrets = self.forward(
            self.model, self.get_infostate_tensor(s), s.legal_actions_mask()
        )
        return regrets

    def regret_matching(self, regrets, legal_actions):
        legal_regrets = np.zeros_like(regrets)
        legal_regrets[legal_actions] = regrets[legal_actions]
        pos_sum = np.sum(np.maximum(legal_regrets, 0))
        if pos_sum > 0:
            return np.maximum(legal_regrets, 0) / pos_sum
        else:
            policy = np.zeros_like(regrets)
            if self.use_regret_matching_argmax:
                max_action_id = legal_actions[np.argmax(regrets[legal_actions])]
                policy[max_action_id] = 1
            else:
                policy[legal_actions] = 1 / len(legal_actions)
            return policy


class AvePolicyTrainer(Trainer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        device: str = "cpu",
        gamma: int = 0,
    ):
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            device=device,
        )
        self.gamma = gamma

    def forward(self, x, mask):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            with torch.no_grad():
                policy = self.predict(x, mask)
                return policy.cpu().numpy()

    def predict(self, x, mask):
        logits = self.model(x)
        legal_logits = torch.where(mask == 1, logits, -10e20)
        policy = self.softmax_fn(legal_logits)
        return policy

    def train_model(self, T):
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] policy loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        return loss.item()

    def compute_loss(self, samples, T):
        infostates, policy, legal_actions_mask, iterations = samples
        iterations = torch.pow(iterations / T * 2, self.gamma / 2)
        outputs = self.predict(infostates, legal_actions_mask)
        loss = self.loss_fn(outputs * iterations, policy * iterations)
        return loss

    def action_probabilities(self, s, probs_as_dict=True):
        policy = self.forward(self.get_infostate_tensor(s), s.legal_actions_mask())
        if probs_as_dict:
            probs_dict = {action: policy[action] for action in s.legal_actions()}
            return probs_dict
        else:
            return policy


class ReservoirBuffer:
    def __init__(self, buffer_size, infostate_size, action_size, device="cpu"):
        self.buffer_size = buffer_size
        self.infostate_size = infostate_size
        self.action_size = action_size
        self.device = device
        self.reset()

    def reset(self):
        if hasattr(self, "cur_id"):
            self.cur_id = 0
            return

        self.infostate_buf = np.ones(
            [self.buffer_size, self.infostate_size], dtype=float
        )
        self.q_value_buf = np.ones([self.buffer_size, self.action_size], dtype=float)
        self.q_value_mask_buf = np.ones(
            [self.buffer_size, self.action_size], dtype=float
        )
        self.iteration_buf = np.ones([self.buffer_size, 1], dtype=float)
        self.cur_id = 0

    def add(self, infostate, q_value, q_value_mask, iteration):
        if self.cur_id < self.buffer_size:
            self.add_data(self.cur_id, infostate, q_value, q_value_mask, iteration)
        else:
            idx = np.random.randint(low=0, high=self.cur_id + 1)
            if idx < self.buffer_size:
                self.add_data(idx, infostate, q_value, q_value_mask, iteration)
        self.cur_id += 1

    def add_data(self, idx, infostate, q_value, q_value_mask, iteration):
        self.infostate_buf[idx] = infostate
        self.q_value_buf[idx] = q_value
        self.q_value_mask_buf[idx] = q_value_mask
        self.iteration_buf[idx] = iteration

    def sample(self, num_samples=-1):
        self.data_length = min(self.cur_id, self.buffer_size)
        if num_samples > self.data_length:
            num_samples = -1
        if num_samples == -1:
            idxs = list(range(self.data_length))
        else:
            idxs = random.sample(range(self.data_length), num_samples)
        data = (
            self.infostate_buf[idxs],
            self.q_value_buf[idxs],
            self.q_value_mask_buf[idxs],
            self.iteration_buf[idxs],
        )
        data_tensor = tuple(map(self.numpy_to_tensor, data))
        return data_tensor

    def numpy_to_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.cur_id


class SonnetLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stddev = 1 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(
            torch.Tensor(
                stats.truncnorm.rvs(
                    -2,
                    2,
                    loc=0,
                    scale=stddev,
                    size=[self.out_features, self.in_features],
                ),
            )
        )
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
        if self.activation:
            y = F.relu(y)
        return y


class ZeroInitLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = nn.Parameter(torch.zeros([self.out_features, self.in_features]))
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
        if self.activation:
            y = F.relu(y)
        return y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        input_sizes = [input_size, *hidden_size[:-1]]
        output_sizes = [*hidden_size]
        activations = [True] * (len(hidden_size))
        layer_list = [
            SonnetLinear(in_size, out_size, activation)
            for in_size, out_size, activation in zip(
                input_sizes, output_sizes, activations
            )
        ]
        layer_list.append(ZeroInitLinear(hidden_size[-1], output_size, False))
        self.layers = nn.Sequential(*layer_list)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        return self.layers(x)
class QValueTrainer(Trainer):
    def __init__(
        self,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        regret_trainers: list[RegretTrainer],
        device: str = "cpu",
    ):
        super().__init__(
            history_size,
            action_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            device,
        )
        self.state_size = state_size
        self.buffer = CircularBuffer(
            self.buffer_size,
            self.input_size,
            state_size,
            self.output_size,
            device=self.device,
        )
        self.model = self.init_model()
        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.regret_trainers = regret_trainers

    def init_model(self):
        model = MLP(self.input_size, self.network_layers, self.output_size).to(
            self.device
        )
        return model

    def get_baseline(self, s: SpielState, player: int) -> np.ndarray:
        history_tensor = self.get_history_tensor(s)
        coef = 1 if player == 0 else -1
        baseline = self.forward(history_tensor) * coef
        baseline = baseline * np.array(s.legal_actions_mask(), dtype=float)
        # baseline =  np.zeros(s.num_distinct_actions(), dtype=float)
        return baseline

    def add_data(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        if done:
            next_state = np.zeros([self.state_size], dtype=float)
            next_legal_actions_mask = np.zeros([self.output_size], dtype=int)
            next_player = 0
        self.buffer.add(
            history,
            action,
            next_history,
            next_state,
            next_legal_actions_mask,
            next_player,
            done,
            reward,
        )

    def train_model(self, T):
        self.model = self.init_model()
        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return
        best_loss = float("inf")
        self.best_model = self.init_model()
        for train_step in range(self.train_steps + 1):
            samples = self.buffer.sample(self.batch_size)
            (
                histories,
                next_histories,
                next_states,
                rewards,
                next_legal_actions_mask,
                next_players,
                actions,
                dones,
            ) = samples

            q_values = self.model(histories)  # [B, A]
            # actions [B]
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            with torch.no_grad():
                next_q_values = self.target_model(next_histories)  # [B, A]

                # calculate next strategies
                players_next_stragies = []
                for player in [0, 1]:
                    p0_regrets = self.regret_trainers[player].predict(
                        self.regret_trainers[player].model,
                        next_states,
                        next_legal_actions_mask,
                    )  # [B, A]
                    p0_legal_regrets = p0_regrets * next_legal_actions_mask  # [B, A]
                    p0_regrets_pos = torch.clamp(p0_regrets, min=0)  # [B, A]
                    p0_regrets_pos_sum = torch.sum(
                        p0_regrets_pos, dim=1, keepdim=True
                    )  # [B, 1]
                    p0_rm_next_strategies = (
                        p0_regrets_pos / p0_regrets_pos_sum
                    )  # [B, A]
                    _, p0_max_legal_action_id = torch.max(
                        torch.where(
                            next_legal_actions_mask == 1,
                            p0_legal_regrets,
                            torch.tensor(float("-inf"), device=self.device),
                        ),
                        dim=1,
                    )  # [B]
                    p0_max_next_strategies = F.one_hot(
                        p0_max_legal_action_id, self.output_size
                    )  # [B, A]
                    p0_next_strategies = torch.where(
                        p0_regrets_pos_sum == 0,
                        p0_max_next_strategies,
                        p0_rm_next_strategies,
                    )  # [B, A]
                    players_next_stragies.append(p0_next_strategies)

                next_strategies = torch.where(
                    next_players.unsqueeze(1) == 0,
                    players_next_stragies[0],
                    players_next_stragies[1],
                )

            target = rewards + (1 - dones) * torch.sum(
                next_q_values * next_strategies, dim=1, keepdim=False
            )

            loss = self.loss_fn(q_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if train_step % 50 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if loss.item() < best_loss:
                best_loss = loss.item()
                self.best_model.load_state_dict(self.model.state_dict())

            if train_step % 100 == 0:
                self.logger.info(
                    "train_step[{}/{}]: loss {}, best_loss {}".format(
                        train_step, self.train_steps, loss.item(), best_loss
                    )
                )

        self.model.load_state_dict(self.best_model.state_dict())
        return best_loss

    def forward(self, x):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return self.model(x).cpu().numpy()


class CircularBuffer:
    def __init__(
        self, buffer_size, history_size, state_size, action_size, device="cpu"
    ):
        self.buffer_size = buffer_size
        self.history_size = history_size
        self.action_size = action_size
        self.state_size = state_size
        self.device = device
        self.reset()

    def reset(self):
        self.history_buf = np.ones([self.buffer_size, self.history_size], dtype=float)
        self.action_buf = np.ones([self.buffer_size], dtype=int)
        self.next_history_buf = np.ones(
            [self.buffer_size, self.history_size], dtype=float
        )
        self.next_state_buf = np.ones([self.buffer_size, self.state_size], dtype=float)
        self.next_legal_actions_mask_buf = np.ones(
            [self.buffer_size, self.action_size], dtype=int
        )
        self.next_player_buf = np.ones([self.buffer_size], dtype=int)
        self.done_buf = np.ones([self.buffer_size], dtype=int)
        self.reward_buf = np.ones([self.buffer_size], dtype=float)
        self.cur_id = 0
        self.size = 0

    def add(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        self.add_data(
            self.cur_id,
            history,
            action,
            next_history,
            next_state,
            next_legal_actions_mask,
            next_player,
            done,
            reward,
        )
        self.cur_id = (self.cur_id + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def add_data(
        self,
        idx,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        self.history_buf[idx] = history
        self.action_buf[idx] = action
        self.next_history_buf[idx] = next_history
        self.next_state_buf[idx] = next_state
        self.next_legal_actions_mask_buf[idx] = next_legal_actions_mask
        self.next_player_buf[idx] = next_player
        self.done_buf[idx] = done
        self.reward_buf[idx] = reward

    def sample(self, num_samples=-1):
        data_length = len(self)
        if num_samples == -1:
            idxs = list(range(data_length))
        else:
            idxs = random.sample(range(data_length), num_samples)
        float_data = (
            self.history_buf[idxs],
            self.next_history_buf[idxs],
            self.next_state_buf[idxs],
            self.reward_buf[idxs],
        )
        int_data = (
            self.next_legal_actions_mask_buf[idxs],
            self.next_player_buf[idxs],
            self.action_buf[idxs],
            self.done_buf[idxs],
        )
        data_tensor = (
            *map(self.numpy_to_float_tensor, float_data),
            *map(self.numpy_to_int_tensor, int_data),
        )
        return data_tensor

    def numpy_to_float_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def numpy_to_int_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.int64, device=self.device)

    def __len__(self):
        return self.size
