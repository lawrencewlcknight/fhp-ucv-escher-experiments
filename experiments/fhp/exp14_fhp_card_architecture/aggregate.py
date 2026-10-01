"""Three-source inference and separate generalisation/gameplay/cost figures."""

import numpy as np
from experiments.fhp.exp13_fhp_policy_capacity.aggregate import aggregate as aggregate_audit, summary
from .config import contract


def aggregate(root, smoke=False, *, config=None):
    config = contract(smoke) if config is None else config
    return aggregate_audit(root, smoke, config=config, plot_fn=plot)


def plot(analysis, config, curves, summaries, values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    architectures = list(config["architectures"])
    budgets = [*config["control_updates"], "tuned"]
    title = "SMOKE ONLY" if config["smoke"] else "FHP card architecture: three source seeds"
    colours = dict(zip(architectures, plt.get_cmap("tab10").colors))

    def finish(fig, name):
        fig.suptitle(title)
        fig.savefig(analysis / f"card_architecture_{name}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, xkey in enumerate(("updates", "fitting_seconds")):
        for row, ykey in enumerate(("training_kl", "validation_kl")):
            ax = axes[row, column]
            for architecture in architectures:
                x, y, errors = [], [], []
                for update in config["screen_updates"]:
                    block = [r for r in curves if r["architecture"] == architecture
                             and r["recipe"] == config["control_recipe"] and r["updates"] == update]
                    seeds = [np.mean([r[ykey] for r in block if r["seed"] == s]) for s in config["seeds"]]
                    x.append(np.mean([r[xkey] for r in block]))
                    y.append(np.mean(seeds)); errors.append(summary(seeds)["se"] or 0)
                ax.errorbar(x, y, yerr=errors, label=architecture, color=colours[architecture], capsize=3, marker=".")
            ax.set(xlabel=xkey.replace("_", " "), ylabel=ykey.replace("_", " "),
                   title="Adam 0.003; bars: one source-seed SE")
            ax.grid(alpha=.2); ax.legend()
    finish(fig, "learning_curves")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, architecture in zip(axes.flat, architectures):
        for recipe in config["recipes"]:
            means = [np.mean([r["validation_kl"] for r in curves if r["architecture"] == architecture
                             and r["recipe"] == recipe and r["updates"] == u]) for u in config["screen_updates"]]
            ax.plot(config["screen_updates"], means, marker=".", label=recipe)
        ax.set(title=architecture, xlabel="Updates", ylabel="Card-held-out validation KL")
        ax.grid(alpha=.2); ax.legend()
    finish(fig, "optimisation")

    metrics = [("diagnostic_test_weighted_kl", "Card-held-out test KL (lower better)"),
               ("lbr_mbb_per_hand", "Restricted LBR gain (lower better)"),
               ("direct_crossplay_archived", "EV vs archived (higher better)"),
               ("direct_crossplay_matched_dense", "EV vs matched dense (higher better)")]
    fig, axes = plt.subplots(len(metrics), len(budgets), figsize=(14, 13), constrained_layout=True)
    for i, (metric, label) in enumerate(metrics):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            for k, architecture in enumerate(architectures):
                arm = f"{architecture}_{budget}"
                if (metric, arm) not in values:
                    continue
                block = list(values[(metric, arm)].values())
                stats = summary(block)
                ax.scatter([k]*len(block), block, alpha=.4, color=colours[architecture])
                ax.errorbar(k, stats["mean"], yerr=stats["se"] or 0, fmt="ko", capsize=4)
            ax.set_xticks(range(len(architectures)), architectures, rotation=25, ha="right")
            ax.set_title(f"{budget}: {label}", fontsize=10)
            if i >= 2:
                ax.axhline(0, color="gray", linewidth=.7)
            ax.grid(axis="y", alpha=.2)
    fig.supxlabel("Points: source means after averaging fits; bars: one source-seed SE. EV and LBR in mbb/hand.")
    finish(fig, "quality")

    lookup = {(r["metric"], r["arm"]): r for r in summaries}
    fig, axes = plt.subplots(2, len(budgets), figsize=(15, 8), constrained_layout=True)
    for i, metric in enumerate(("fitting_seconds", "inference_batch_1_seconds_per_state")):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            for architecture in architectures:
                arm = f"{architecture}_{budget}"
                cost, quality = lookup[(metric, arm)], lookup[("lbr_mbb_per_hand", arm)]
                ax.errorbar(cost["mean"], quality["mean"], xerr=cost["se"] or 0, yerr=quality["se"] or 0,
                            fmt="o", color=colours[architecture], label=architecture)
            ax.set(title=str(budget), xlabel=metric.replace("_", " "), ylabel="Restricted LBR gain (mbb/hand)")
            ax.grid(alpha=.2); ax.legend()
    finish(fig, "cost_quality")
