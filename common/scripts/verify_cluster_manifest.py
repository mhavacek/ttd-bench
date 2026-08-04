#!/usr/bin/env python3
"""Verify that the local copy of the frozen runs matches the cluster bit for bit.

  python common/scripts/verify_cluster_manifest.py manifest-frozen.md5

The manifest is produced on the cluster, inside the OLD repo layout:

  cd .../ttd-bench && find results-scvd results-phase1-final -type f \
      -exec md5sum {} + | sort -k2 > ~/manifest-frozen.md5

Cluster paths are remapped onto the restructured local tree (see PATH_MAP).
Exit status is 0 only when both directions are clean: nothing missing, nothing
mismatched. Extra local files are reported but do not fail the check — the
local tree legitimately gains derived files the cluster never had.

Nothing is deleted anywhere; this only reports.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # common/
ROOT = REPO.parent                                     # branch-A root

# cluster (old layout)  ->  local (restructured)
PATH_MAP = {
    "results-scvd/": "results/scvd-phase2/",
    "results-phase1-final/": "results/phase1-final/",
}


def remap(cluster_path: str) -> str | None:
    for old, new in PATH_MAP.items():
        if cluster_path.startswith(old):
            return new + cluster_path[len(old):]
    return None


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="manifest-frozen.md5 fetched from the cluster")
    ap.add_argument("--root", default=str(ROOT), help="branch-A root (default: inferred)")
    args = ap.parse_args()
    root = Path(args.root)

    expected: dict[str, str] = {}
    unmapped: list[str] = []
    for line in Path(args.manifest).read_text().splitlines():
        if not line.strip():
            continue
        # `md5sum` output: "<hash>  <path>" (two spaces); paths may contain spaces
        digest, _, cluster_path = line.partition("  ")
        local = remap(cluster_path.strip())
        if local is None:
            unmapped.append(cluster_path.strip())
        else:
            expected[local] = digest.strip()

    print(f"manifest: {len(expected)} souboru z klastru")
    if unmapped:
        print(f"  !! {len(unmapped)} cest se nepodarilo premapovat, napr.:")
        for p in unmapped[:5]:
            print(f"       {p}")

    missing, mismatch, ok = [], [], 0
    for rel, digest in sorted(expected.items()):
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif md5(p) != digest:
            mismatch.append(rel)
        else:
            ok += 1

    # the other direction: local files under the mapped roots that the cluster lacks
    extra = []
    for new in PATH_MAP.values():
        base = root / new
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(root))
                if rel not in expected:
                    extra.append(rel)

    print()
    print(f"  shoduje se     : {ok}")
    print(f"  CHYBI lokalne  : {len(missing)}")
    print(f"  NESEDI otisk   : {len(mismatch)}")
    print(f"  navic lokalne  : {len(extra)}  (odvozene soubory, neni chyba)")

    for label, items in (("CHYBI", missing), ("NESEDI", mismatch)):
        if items:
            print(f"\n  --- {label} ({len(items)}) ---")
            for rel in items[:40]:
                print(f"      {rel}")
            if len(items) > 40:
                print(f"      … a dalsich {len(items) - 40}")

    clean = not missing and not mismatch and not unmapped
    print()
    print("VYSLEDEK:", "OK — lokalni kopie je uplna a bit-shodna, na klastru se smi mazat"
          if clean else "NEPROSLO — na klastru NEMAZAT nic")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
