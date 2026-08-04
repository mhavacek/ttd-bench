#!/usr/bin/env python3
"""Inter-annotator agreement for t_onset annotations (SCVD scenarios).

Two modes:

1) --make-blind: from the primary annotator's YAML, produce a BLIND template
   for the second annotator — a random half (seed 42) of the USABLE scenarios,
   with onsets and usability flags stripped, in shuffled order. Ship it with
   the same annotate_onsets.py UI the student used.

     python scripts/annotation_agreement.py --make-blind \
         --primary configs/scenarios-scvd.yaml --out scenarios-scvd-blind.yaml

2) --compare: after the second annotator returns their YAML, compute
   agreement on the doubly-annotated scenarios:
     - usability agreement (both said usable/unusable)
     - onset difference in frames and ms (mean |d|, median |d|, max |d|,
       share within +-3 / +-5 / +-10 frames)

     python scripts/annotation_agreement.py --compare \
         --primary configs/scenarios-scvd.yaml --second scenarios-scvd-blind.yaml

The +-X-frame agreement figure goes into the paper's methodology ("onset
ground truth was double-annotated for N scenarios; inter-annotator agreement
was ... frames"), and mean |d| bounds the annotation component of TTD
measurement uncertainty.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml


def load(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_blind(primary_path: str, out_path: str, fraction: float, seed: int) -> int:
    src = load(primary_path)
    scen = {s["id"]: s for s in src["scenarios"]}
    usable = [sid for sid in src.get("active_scenarios", []) if sid in scen]
    if not usable:
        print("no active (usable) scenarios in primary YAML", file=sys.stderr)
        return 1

    rng = random.Random(seed)
    n_pick = max(1, round(len(usable) * fraction))
    picked = rng.sample(sorted(usable), n_pick)
    rng.shuffle(picked)  # shuffled order: no signal from file position

    blind = []
    for sid in picked:
        s = dict(scen[sid])
        s["t_onset_frame"] = None
        tags = dict(s.get("tags", {}))
        tags["onset_verified"] = False
        tags["unusable"] = False
        s["tags"] = tags
        blind.append(s)

    out = {
        "_note": ("BLIND double-annotation set (second annotator). Annotate "
                  "t_onset_frame with annotate_onsets.py exactly like the "
                  "primary pass; do not consult the primary annotations."),
        "scenarios": blind,
        "active_scenarios": [],
    }
    Path(out_path).write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"blind template: {n_pick}/{len(usable)} usable scenarios -> {out_path}")
    return 0


def compare(primary_path: str, second_path: str) -> int:
    prim = {s["id"]: s for s in load(primary_path)["scenarios"]}
    sec = {s["id"]: s for s in load(second_path)["scenarios"]}
    common = sorted(set(prim) & set(sec))
    if not common:
        print("no common scenario ids", file=sys.stderr)
        return 1

    diffs_frames: list[float] = []
    usab_agree = 0
    usab_total = 0
    rows = []
    for sid in common:
        p, s = prim[sid], sec[sid]
        p_use = bool(p.get("tags", {}).get("onset_verified")) and not p.get("tags", {}).get("unusable")
        s_use = bool(s.get("tags", {}).get("onset_verified")) and not s.get("tags", {}).get("unusable")
        usab_total += 1
        if p_use == s_use:
            usab_agree += 1
        if p_use and s_use and p.get("t_onset_frame") is not None and s.get("t_onset_frame") is not None:
            d = s["t_onset_frame"] - p["t_onset_frame"]
            diffs_frames.append(d)
            fps = float(p.get("fps", 30.0))
            rows.append((sid, p["t_onset_frame"], s["t_onset_frame"], d, 1000.0 * d / fps))

    print(f"doubly-annotated scenarios: {usab_total}")
    print(f"usability agreement:        {usab_agree}/{usab_total} "
          f"({100.0 * usab_agree / usab_total:.0f} %)")
    if not diffs_frames:
        print("no scenarios with onsets from both annotators")
        return 0

    absd = [abs(d) for d in diffs_frames]
    absd_sorted = sorted(absd)
    n = len(absd)
    fps_ref = 30.0
    print(f"\nonset agreement (n={n}):")
    print(f"  mean |d|   = {sum(absd)/n:.2f} frames  ({1000*sum(absd)/n/fps_ref:.1f} ms @30fps)")
    print(f"  median |d| = {absd_sorted[n//2]:.1f} frames")
    print(f"  max |d|    = {max(absd):.0f} frames")
    for k in (3, 5, 10):
        share = sum(1 for d in absd if d <= k) / n
        print(f"  within +-{k:2d} frames: {100*share:.0f} %")
    print(f"  signed mean d (bias) = {sum(diffs_frames)/n:+.2f} frames")

    print("\nper-scenario (id, primary, second, d_frames, d_ms):")
    for r in sorted(rows, key=lambda x: -abs(x[3])):
        print(f"  {r[0]:20s} {r[1]:5d} {r[2]:5d} {r[3]:+5d}  {r[4]:+7.1f} ms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-blind", action="store_true")
    mode.add_argument("--compare", action="store_true")
    ap.add_argument("--primary", required=True, help="primary annotator's YAML")
    ap.add_argument("--second", help="second annotator's YAML (--compare)")
    ap.add_argument("--out", default="scenarios-scvd-blind.yaml", help="--make-blind output")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="share of usable scenarios to double-annotate (default 0.5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.make_blind:
        return make_blind(args.primary, args.out, args.fraction, args.seed)
    if not args.second:
        ap.error("--compare requires --second")
    return compare(args.primary, args.second)


if __name__ == "__main__":
    sys.exit(main())
