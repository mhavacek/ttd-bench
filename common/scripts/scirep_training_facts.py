"""Training facts of the four Phase-2 checkpoints, for the Sci Rep Methods.

Reads the archived Ultralytics runs (archiv/Data-zenodo/run_*) and the
checkpoints the campaign used, and writes results/scirep-stats/
training_facts.txt, which scirep_build.py audits the manuscript against.

- stop epoch: last row of results_*.csv (early stopping, patience in args.yaml)
- batch: args.yaml says 32 for every run; the checkpoint's own train_args are
  read from its pickle, because YOLOv12m records 16 there
- class instances: Ultralytics prints them only into labels.jpg (bar labels
  2178 Knife, 2283 Handgun, read by eye in run_8_m; identical data.yaml for
  all runs), so they are entered here by hand
"""
import csv
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "archiv/Data-zenodo"
CKPT = ROOT / "clanek-2-ttdbench/checkpoints"
OUT = ROOT / "results/scirep-stats/training_facts.txt"
MODELS = {"8_s": "yolov8s", "8_m": "yolov8m", "12_m": "yolov12m", "26_m": "yolov26m"}
INSTANCES = {"Knife": 2178, "Handgun": 2283}  # labels.jpg, see docstring
# Kaggle data card and dataset.yaml of raghavnanjappan/weapon-dataset-for-yolov5
# (v2), confirmed by L.K.'s thesis, section 5.1
IMAGES = {"train": 4000, "val": 156}


def ckpt_batch(pt: Path) -> int:
    z = zipfile.ZipFile(pt)
    d = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    seg = d[d.find(b"train_args"):]
    m = re.search(rb"batchr?.{0,6}?K(.)", seg, re.S)  # pickle BININT1
    return m.group(1)[0]


lines = ["model\targs_epochs\targs_patience\targs_batch\tckpt_batch\tstop_epoch"]
for run, name in MODELS.items():
    args = dict(re.findall(r"^(\w+): (.*)$", (RUNS / f"run_{run}/args.yaml").read_text(), re.M))
    rows = list(csv.reader((RUNS / f"run_{run}/results_{run}.csv").open()))
    stop = int(float(rows[-1][0]))
    lines.append(f"{name}\t{args['epochs']}\t{args['patience']}\t{args['batch']}"
                 f"\t{ckpt_batch(CKPT / f'{name}-weapon.pt')}\t{stop}")
lines.append("")
lines += [f"instances_{k}\t{v}\t{v:,}" for k, v in INSTANCES.items()]
lines += [f"images_{k}\t{v}\t{v:,}" for k, v in IMAGES.items()]
OUT.write_text("\n".join(lines) + "\n")
print(OUT.read_text())
