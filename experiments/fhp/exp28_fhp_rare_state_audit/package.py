"""Package all diagnostic JSON (not weights) into the conventional analysis prefix."""
import argparse
from pathlib import Path
import shutil
from .common import verify, complete, fresh


def package(root, output):
    root=Path(root)
    success=verify(root)
    output=fresh(output)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".json",".md"):
            relative=path.relative_to(root)
            if relative.parts[0]=="analysis":
                relative=Path(*relative.parts[1:])
            elif path.name=="SUCCESS.json":
                # Source manifest hashes include omitted model binaries.
                relative=relative.with_name("SOURCE_SUCCESS.json")
            target=output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    complete(output,experiment=28,smoke=success["smoke"],local_smoke=success["local_smoke"],
             commit=success["commit"],seeds=success["seeds"],scope="analysis-only; full candidate weights under audit/")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    package(args.root,args.output)
