from pathlib import Path


def test_no_obsolete_variant_references_or_outputs():
    root = Path(__file__).resolve().parents[1]
    obsolete_name = "le" + "duc"
    excluded = {".git", ".pytest_cache", "__pycache__", "outputs"}
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
        if obsolete_name in text.lower() or obsolete_name in path.name.lower():
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
