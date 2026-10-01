"""Output-only retention for the new FHP runs; historical jobs are unchanged."""


def apply_retention(job, *, kind, final_state=False):
    spec = job["taskGroups"][0]["taskSpec"]
    script = spec["runnables"][0]["script"]
    if kind == "train":
        # Without intermediate states a retry cannot continue training. Require
        # an explicit new run after interruption instead of silently paying twice.
        spec["maxRetryCount"] = 0
        script["text"] = script["text"].replace(
            'elif gcloud storage ls "$REMOTE_TASK/training_states/**"',
            'elif gcloud storage ls "$REMOTE_TASK/run_manifest.json"')
    if kind != "controller":
        excluded = r"(^|/)continuation_inputs(/|$)|\.tmp$"
        if not final_state or kind == "smoke":
            excluded += r"|(^|/)training_states(/|$)|\.pt$"
        script["text"] = script["text"].replace(
            'gcloud storage rsync --recursive "',
            f'gcloud storage rsync --recursive --exclude=\'{excluded}\' "')
    return job
