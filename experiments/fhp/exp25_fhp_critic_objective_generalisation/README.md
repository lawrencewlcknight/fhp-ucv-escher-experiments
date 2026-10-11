# Experiment 25 — critic objective generalisation

This secondary study uses the shared [Experiment 24 workflow](../exp24_fhp_critic_budget_stages/README.md).
It needs **no second reference training run and no separate Batch submission**.
Use its `smoke-cloud`, then `run` command; both experiments' reports are produced.

At the identical pre-fit 1h, 6h and historical 24h-plus-one-iteration states,
compare native replay TD (practical reference), fresh-panel ordinary TD,
sampling-aware weighted TD, and the recursive corrected-regret-variance objective.
Record 0/100/500/1000/2500/5000/10000-update prefixes. Student critic architecture,
weights, Adam, target history, policies/proposals and source replay are fixed.
The three fresh-data arms share identical traces and minibatch indices; native
replay TD intentionally uses its original replay and batch size.

For a sampled action A with positive probability xi[A], the estimator is

```
qhat = b + onehot(A) * (G - b[A]) / xi[A]
rhat = legal_mask * (qhat - sum(policy * qhat))
loss_variance = ||rhat_1 - rhat_2||^2 / (2 * number_of_legal_actions)
```

The correction is applied **recursively**, including descendants, before root
centring. The two independent traces start at the same full decision history.
In expectation this is mean per-action conditional variance, including the
covariance induced by centring. It does not penalise different true action values
or the conditional mean regret. Frozen policies/proposals and full support are
maintained; candidate baselines cannot retroactively choose sampled actions.
Existing exact-tree tests verify the unbiasedness and paired-variance identity.

Weighted TD uses `1/(legal_action_count * xi[A])`, a straightforward
sampling-aware root-action weighting, **not** a claim of a variance-optimal
baseline. Fresh TD labels use the old frozen target. All methods optimise their
online critic; deployment uses the native four-fit parameter average, not output
averaging. Its zero-update control performs no deployment refresh.

At every prefix, measure the **entire fixed fitting panel** (all objectives and
corrected-regret variance), a fixed native replay TD probe, and independent
selection/assessment panels for both online and deployed critics. The replay
probe and fitting panel are in-sample diagnostics, not budget-selection data.
This distinguishes falling empirical loss with worsening generalisation from
failure to reduce the fitting objective, and exposes online/deployment mismatch.
These explanations may coexist; finite curves alone do not prove causality.
Training-panel risk is never used to reselect candidates after assessment.

The same finite 4,096-pair fitting panel per member is sampled with replacement:
2,500/5,000/10,000 steps represent approximately 20/39/78 panel passes. The new
100/500/1,000 prefixes test the earlier regime. Any subsequent change in panel
size, refresh strategy or learning rate needs matched TD controls; it is not
silently included in this study. No broad hyperparameter search is launched.

Report complete fit/selection/test curves, tails and total cost. A favourable
independent deployed-critic result warrants replication, not an automatic
full-training promotion. Failure shows no demonstrated benefit for these states
and this tested protocol, not impossibility of alternative objectives in general.
One seed, two new stages and one historical late stage do not establish generality,
adaptive allocation superiority, lower exploitability or stronger poker play.
