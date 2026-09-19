from pathlib import Path


def test_no_obsolete_variant_references_or_outputs():
    root = Path(__file__).resolve().parents[1]
    obsolete_name = "le" + "duc"
    excluded = {".git", ".pytest_cache", "__pycache__", "outputs"}
    intentional_provenance = {
        "README.md",
        "docs/OUTPUT_CONVENTIONS.md",
        "docs/THESIS_ARTIFACTS.md",
        "experiments/fhp/exp3_archived_ucv_escher_parallel/config.py",
        "experiments/fhp/exp1_fhp_grouped_wide_ucv_baseline/README.md",
        "experiments/fhp/exp1_fhp_grouped_wide_ucv_baseline/config.py",
        "tests/test_exp1_grouped_wide_baseline.py",
        "unbiased_escher/PARALLEL_UPSTREAM.md",
        "unbiased_escher/grouped_wide_solver.py",
        "unbiased_escher/parallel_solver.py",
    }
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pkl", ".png", ".jpg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = str(path.relative_to(root))
        if (
            obsolete_name in text.lower() or obsolete_name in path.name.lower()
        ) and relative not in intentional_provenance:
            offenders.append(relative)
    assert offenders == []
