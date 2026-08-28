# Testing

Run the unit suite and the end-to-end one-iteration smoke test with:

```bash
python -m pytest
python -m experiments.fhp.exp1_ucv_escher_baseline.run --smoke
```

The suite verifies the exact OpenSpiel FHP parameters, the transferred UCV
configuration, checkpoint reloadability, and absence of obsolete variant
artifacts.
