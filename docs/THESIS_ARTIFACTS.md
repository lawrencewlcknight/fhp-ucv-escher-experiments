# Thesis artifact curation

Complete FHP run directories are working data. They contain reloadable policy
checkpoints, logs, resource traces, and possibly failure evidence, so
`outputs/` and `cloud_outputs/` remain ignored by git.

The Leduc ESCHER repository has a mature promotion script and tracked
`thesis_artifacts/` tree. This FHP repository does not yet implement that
promotion pipeline. Do not invoke or copy the Leduc script blindly: FHP uses a
different artifact contract, time-based checkpoints, and no exact
exploitability curves.

## Preserve the source run first

Before curating results:

1. download the complete Cloud Storage output, including diagnostics;
2. verify both checkpoint SHA-256 values against the manifests;
3. confirm that `summary.json` records `training_time_budget` and two
   checkpoints;
4. retain the immutable source output outside git;
5. perform sampled policy evaluation from the verified checkpoints.

Failed or partial runs may be useful forensic records, but they must not be
promoted as completed experiment results.

## Lightweight artifacts suitable for future promotion

When a tracked FHP thesis-artifact tree is introduced, promote only small,
reviewable outputs such as:

- plots (`*.png` or publication-ready vector equivalents);
- aggregate and checkpoint comparison tables (`*.csv`);
- sampled head-to-head summaries (`*.json` or `*.csv`);
- `run_manifest.json` and selected provenance metadata;
- `summary.json` and capacity-assessment results;
- a promotion manifest recording source job, source commit, selected files,
  hashes, and promotion timestamp.

Keep these heavyweight or diagnostic artifacts out of the tracked tree:

- `*.pkl`, `*.pt`, `*.pth`, and `*.npz` model or array files;
- replay storage and snapshots;
- `*.log` and resource snapshot streams;
- raw per-hand head-to-head traces;
- failed-run tracebacks unless a specific forensic appendix requires them.

## Suggested future layout

Use the same high-level convention as the Leduc repository while retaining the
numbered FHP experiment identity:

```text
thesis_artifacts/
  expN_experiment_name/
    source_run_directory_name/
      promotion_manifest.json
      run_manifest.json
      summary.json
      figures-and-tables...
```

A future promotion utility should default to dry-run, refuse overwrites unless
explicitly requested, filter heavyweight files, and never push from a Batch VM.
Promotion should remain a local review step so outputs can be inspected with
`git diff` before they enter version control.
