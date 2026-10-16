import glob
import json
import re
import os

GROUP_RE = re.compile(r"^(\d{2}-\d{2}-\d{2})_(prone|supine)_(.+)$")

def parse_group(group):
    m = GROUP_RE.match(os.path.basename(group.rstrip("/")))
    return m.groups() if m else (None, None, None)

# Returns the mimolab experiment (json dict)
def get_experiment(date, haltung, suffix, is_cee=False, is_dcee=False):
    if is_cee and is_dcee:
        raise ValueError("Specify at most one of 'is_cee' and 'is_dcee'.")
        return None
    for path in glob.glob(os.path.expanduser("~/MIMo/.mimolab/evals/*.json")):
        filename = os.path.basename(path)

        if is_cee and not filename.startswith("cee"):
            continue
        elif is_dcee and not filename.startswith("dcee"):
            continue
        elif not is_dcee and not is_cee and not filename.startswith("group"):
            continue

        payload = json.load(open(path))
        _date, _haltung, _suffix = parse_group(payload["group"])
        if _date is None: continue
        if _date != date: continue
        if _haltung != haltung: continue
        if _suffix != suffix: continue
        return payload
    raise ValueError(f"Run date {date}, haltung {haltung}, suffix {suffix} does not exist!")
    return None

# Returns the experiments in supine / prone with or without
# metabolic cost penalization.
def get_baseline_experiments(is_prone, is_metabolicpen, is_cee=False, is_dcee=False):
    payloads = []

    if is_prone:
        if is_metabolicpen:
            payloads.append(get_experiment('26-10-08', 'prone', 'age1_metabolicpen34', is_cee, is_dcee))
            payloads.append(get_experiment('26-10-08', 'prone', 'age3_metabolicpen34', is_cee, is_dcee))
            payloads.append(get_experiment('26-10-07', 'prone', 'age6_metabolicpen34', is_cee, is_dcee))
            payloads.append(get_experiment('26-10-07', 'prone', 'age9_metabolicpen34', is_cee, is_dcee))
        else:
            payloads.append(get_experiment('26-09-13', 'prone', 'age1', is_cee, is_dcee))
            payloads.append(get_experiment('26-03-10', 'prone', 'age3', is_cee, is_dcee))
            payloads.append(get_experiment('26-05-19', 'prone', 'age6', is_cee, is_dcee))
            payloads.append(get_experiment('26-03-10', 'prone', 'age9', is_cee, is_dcee))
    else:
        if is_metabolicpen:
            payloads.append(get_experiment('26-10-08', 'supine', 'age1_metabolicpen34', is_cee, is_dcee))
            payloads.append(get_experiment('26-10-12', 'supine', 'age3_metabolicpen34', is_cee, is_dcee))
            payloads.append(get_experiment('26-10-08', 'supine', 'age6_metabolicpen34', is_cee, is_dcee))
            payloads.append(get_experiment('26-10-11', 'supine', 'age9_metabolicpen34', is_cee, is_dcee))
        else:
            payloads.append(get_experiment('26-03-09', 'supine', 'age1', is_cee, is_dcee))
            payloads.append(get_experiment('26-03-09', 'supine', 'age3', is_cee, is_dcee))
            payloads.append(get_experiment('26-05-18', 'supine', 'age6', is_cee, is_dcee))
            payloads.append(get_experiment('26-03-07', 'supine', 'age9', is_cee, is_dcee))

    return payloads
