"""Stdlib-only cloud smoke gate; a synthetic smoke cannot authorise the full job."""
import argparse
import hashlib
import json
import subprocess


def validate(success_raw, summary_raw, commit):
    success,summary=json.loads(success_raw),json.loads(summary_raw)
    if (success.get("status")!="complete" or success.get("experiment")!=28
            or success.get("smoke") is not True or success.get("local_smoke") is not False
            or success.get("commit")!=commit or success.get("seeds")!=[0]
            or success.get("artifacts",{}).get("analysis/summary.json")!=hashlib.sha256(summary_raw).hexdigest()):
        raise ValueError("Required real cloud smoke, source commit or report checksum differs")
    if summary.get("seeds")!=[0] or summary.get("pilot") is not False or len(summary.get("per_seed",[]))!=1:
        raise ValueError("Incomplete smoke workflow")
    row=summary["per_seed"][0]
    if (row["continuation"]["ray_actors"]!=8 or row["continuation"]["complete_iterations"]!=1
            or row["continuation"]["synthetic"] or row["continuation"]["diagnostic_samples_added"]!=0
            or not row["gameplay"]["complete"] or not row["gameplay"]["rows"]):
        raise ValueError("Smoke omitted native restore, eight actors or gameplay")
    for hour,run in (("24","exp10-features-20261001-161740"),("48","exp16-feat48-20261004-182051")):
        h=row["historical"][hour]["summary"]
        if h["synthetic"] or h["source"]["run"]!=run or not h["estimator_expectation_verified"]:
            raise ValueError("Wrong historical source or missing estimator check")
    fit=row["historical"]["48"]["summary"]["refitting"]
    if not fit["source_preserved"] or not fit["assessment_locked"] or fit["candidates"]<2:
        raise ValueError("Smoke omitted refitting/assessment integrity")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri",required=True)
    p.add_argument("--commit",required=True)
    args=p.parse_args()
    raw=[subprocess.check_output(["gcloud","storage","cat",args.uri.rstrip("/")+"/audit/"+name])
         for name in ("SUCCESS.json","analysis/summary.json")]
    validate(*raw,args.commit)


if __name__=="__main__":
    main()
