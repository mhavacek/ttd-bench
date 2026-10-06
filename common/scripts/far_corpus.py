#!/usr/bin/env python3
"""Build the FAR-calibration corpus manifest and prove disjointness from Phase 2.

The corpus is every clip of the SCVD *Normal* class (SCVD_converted, Train +
Test; the sec_split tree holds the same footage re-cut into 1 s pieces and is
excluded as duplicate material), plus — since 2026-08-10 — the UCF-Crime
normal split, which was fetched precisely because 0.2728 h could not resolve
a false-alarm rate below 12/h.

UCF-Crime normal clips are alarm-free by dataset definition (the "Normal"
class is the negative class of the anomaly benchmark) and come from the same
CCTV domain, but they are longer and more varied than SCVD Normal, so the
manifest keeps a `source` column and every rate can be reported per source.

Disjointness from the 34 Phase-2 event clips is checked three ways:
  1. by construction — event clips come from the Weaponized class folders,
     the corpus only from Normal class folders;
  2. by filename — no basename appears in both sets;
  3. by content — no MD5 checksum appears in both sets.

The event clips' checksums are written out as event_clips_md5.csv so that
check 3 can be re-run later on a machine that does not hold the video files:
once the corpus is scored, the manifest and that file contain everything the
report needs, and --from-manifest rebuilds it without touching a single video.

Output: results-far-calib/corpus_manifest.csv + event_clips_md5.csv +
disjointness_report.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent   # branch-A root
# Dataset root differs per machine: ~/Datasets/ttd-bench on the dev Mac,
# but on the cluster $HOME is a different storage than the data, so it must
# be passed in (--data-root). Nothing here may assume they coincide.
DEFAULT_DATA_ROOT = Path.home() / "Datasets/ttd-bench"
# Kept as separate sources so a corpus can be scored in stages: the designated
# testing-normal split first, the training splits when there is GPU time.
UCF_SUBDIRS = {
    "ucf-test": "ucf-crime/normal-testing",
    "ucf-train1": "ucf-crime/normal-train-1",
    "ucf-train2": "ucf-crime/normal-train-2",
}
SCEN = ROOT / "clanek-2-ttdbench/configs/scenarios-scvd.yaml"
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"
MANIFEST = OUT / "corpus_manifest.csv"
EVENT_MD5 = OUT / "event_clips_md5.csv"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(p: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def relocate(p: Path, data_root: Path) -> Path:
    """Map an absolute path from a frozen config onto this machine's data root.

    Scenario files carry absolute paths that were rewritten for whichever
    machine last ran them. The disjointness check needs the files themselves,
    so a path that does not exist is retried under data_root, anchored on the
    dataset directory name.
    """
    if p.exists():
        return p
    parts = p.parts
    for anchor in ("scvd", "SCVD", "ucf-crime"):
        if anchor in parts:
            i = parts.index(anchor)
            cand = data_root.joinpath(*parts[i:])
            if cand.exists():
                return cand
            # dev-Mac layout nests SCVD under scvd/
            cand = data_root / "scvd" / Path(*parts[i:])
            if cand.exists():
                return cand
    return p


def build_report(rows: list[dict], ev_rows: list[dict],
                 missing_ev: list[str], ev_folder_ok: bool
                 ) -> tuple[list[str], bool]:
    """Assemble the disjointness report from manifest rows and event checksums.

    Kept separate from the filesystem walk so that --from-manifest produces a
    byte-identical report from the cached CSVs: a report that can only be
    regenerated on the machine holding 94 h of video is a report nobody
    regenerates, and it goes stale exactly as this one did.
    """
    ev_names = {r["clip"] for r in ev_rows}
    ev_md5 = {r["md5"] for r in ev_rows if r["md5"]}
    co_names = {r["clip"] for r in rows}
    co_md5 = {r["md5"] for r in rows}
    name_overlap = ev_names & co_names
    md5_overlap = ev_md5 & co_md5
    folder_ok = all("/Normal/" in r["path"] or r["source"].startswith("ucf")
                    for r in rows)

    total: dict[str, float] = {}
    for r in rows:
        total[r["source"]] = total.get(r["source"], 0.0) + float(r["duration_s"])
    hours = sum(total.values()) / 3600

    # Duplicates inside the corpus. UCF-Crime ships the same file under
    # different names across its splits, so a clip can enter the corpus twice
    # and contribute its false alarms twice. That is not extra evidence, and
    # the resolution limits below are therefore computed on unique footage.
    seen: dict[str, dict] = {}
    dup_pairs, dup_seconds = [], 0.0
    for r in sorted(rows, key=lambda r: (r["source"], r["clip"])):
        first = seen.get(r["md5"])
        if first is None:
            seen[r["md5"]] = r
        else:
            dup_pairs.append((first["clip"], first["source"],
                              r["clip"], r["source"], float(r["duration_s"])))
            dup_seconds += float(r["duration_s"])
    uniq_hours = hours - dup_seconds / 3600

    rep = [
        f"corpus clips: {len(rows)} (" + ", ".join(
            f"{src} {sum(r['source'] == src for r in rows)}"
            for src in sorted({r["source"] for r in rows})) + ")",
        f"event clips (phase-2 active_scenarios): {len(ev_rows)}",
        f"event clips missing on disk: {missing_ev or 'none'}",
        f"check 1 (folders): corpus all under Normal/: {folder_ok}; "
        f"event all under Weaponized/: {ev_folder_ok}",
        f"check 2 (basenames): overlap = {sorted(name_overlap) or 'NONE'}",
        f"check 3 (md5): overlap = {sorted(md5_overlap) or 'NONE'}",
        "",
        # enumerate every source present, never a hand-listed subset: an
        # unlisted source silently vanishes from the breakdown while still
        # counting in the total, and the two stop adding up
        *[f"footage {src:<12} {total.get(src, 0) / 3600:9.4f} h "
          f"({total.get(src, 0):.1f} s)" for src in sorted(total)],
        f"footage total:       {hours:.4f} h ({sum(total.values()):.1f} s)",
        "",
        f"duplicate clips inside the corpus: {len(dup_pairs)} "
        f"({dup_seconds / 3600:.4f} h = {100 * dup_seconds / 3600 / hours:.2f} % "
        f"of the total)",
        *[f"  {a} ({sa}) == {b} ({sb})  {d:.2f} s"
          for a, sa, b, sb, d in dup_pairs],
        f"unique footage: {uniq_hours:.4f} h",
        "",
        "resolution limits (Garwood, 95 % one-sided, on unique footage):",
        f"  zero observed episodes supports FAR < {3 / uniq_hours:.3f}/h",
        f"  +-50 % precision (~16 episodes) needs FAR >= {16 / uniq_hours:.3f}/h",
    ]
    ok = folder_ok and ev_folder_ok and not name_overlap and not md5_overlap
    return rep, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="dataset root holding scvd/ and ucf-crime/")
    ap.add_argument("--from-manifest", action="store_true",
                    help="rebuild the report from corpus_manifest.csv and "
                         "event_clips_md5.csv without reading any video")
    args = ap.parse_args()

    if args.from_manifest:
        rows = list(csv.DictReader(open(MANIFEST)))
        ev_rows = list(csv.DictReader(open(EVENT_MD5)))
        missing_ev = [r["path"] for r in ev_rows if not r["md5"]]
        ev_folder_ok = all("/Weaponized/" in r["path"] for r in ev_rows)
        rep, ok = build_report(rows, ev_rows, missing_ev, ev_folder_ok)
        (OUT / "disjointness_report.txt").write_text("\n".join(rep) + "\n")
        print("\n".join(rep))
        print("\nDISJOINT:", "YES" if ok else "NO — STOP")
        return 0 if ok else 1

    data_root = Path(args.data_root)
    SCVD = data_root / "scvd/SCVD/SCVD_converted"
    UCF_NORMAL = {k: data_root / v for k, v in UCF_SUBDIRS.items()}
    print(f"data root: {data_root}")

    scen = yaml.safe_load(open(SCEN))
    active = set(scen["active_scenarios"])
    event = {s["id"]: relocate(Path(s["file"]), data_root)
             for s in scen["scenarios"] if s["id"] in active}
    assert len(event) == 34, f"expected 34 event clips, got {len(event)}"

    corpus = [(p, "scvd") for p in sorted((SCVD / "Train/Normal").glob("*.avi"))]
    corpus += [(p, "scvd") for p in sorted((SCVD / "Test/Normal").glob("*.avi"))]
    for src, d in UCF_NORMAL.items():
        corpus += [(p, src) for p in sorted(d.rglob("*.mp4"))]

    OUT.mkdir(parents=True, exist_ok=True)
    rows, total = [], {}
    for p, source in corpus:
        fps, n = probe(p)
        dur = n / fps if fps else 0.0
        split = p.parent.parent.name.lower() if source == "scvd" else "normal"
        if fps <= 0 or n <= 0:
            print(f"SKIP unreadable: {p}")
            continue
        rows.append({"clip": p.name, "source": source, "split": split,
                     "path": str(p), "fps": round(fps, 3), "n_frames": n,
                     "duration_s": round(dur, 2), "md5": md5(p)})
        total[source] = total.get(source, 0.0) + dur

    with open(OUT / "corpus_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    ev_md5 = {sid: md5(q) for sid, q in event.items() if q.exists()}
    with open(EVENT_MD5, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "clip", "path", "md5"])
        for sid, q in sorted(event.items()):
            w.writerow([sid, q.name, str(q), ev_md5.get(sid, "")])

    ev_rows = [{"scenario_id": sid, "clip": q.name, "md5": ev_md5.get(sid, "")}
               for sid, q in sorted(event.items())]
    missing_ev = [str(q) for q in event.values() if not q.exists()]
    ev_folder_ok = all("/Weaponized/" in str(q) for q in event.values())
    rep, ok = build_report(rows, ev_rows, missing_ev, ev_folder_ok)
    (OUT / "disjointness_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))

    print("\nDISJOINT:", "YES" if ok else "NO — STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
