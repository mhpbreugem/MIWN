#!/usr/bin/env python3
"""
stale.py — flag solutions built against an out-of-date methods pin.

Reads the current Standards submodule commit (the pin). Then:
  - any solutions/pool/<problem>/<vNNNN>/meta.json whose standards_methods_sha
    != pin is reported STALE;
  - any solutions/by-tex/<stem>/lock.json that pins a STALE version (or records a
    standards_methods_sha != pin) is reported AFFECTED.

releases/ is intentionally skipped (frozen snapshots pin older methods on purpose).

  python3 scripts/stale.py            # report only (exit 0)
  python3 scripts/stale.py --strict   # exit nonzero if anything is stale  (CI gate)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "solutions" / "pool"
BYTEX = REPO / "solutions" / "by-tex"


def methods_pin() -> str:
    """Current Standards submodule commit (the pin)."""
    sub = REPO / "standards"
    try:
        return subprocess.run(
            ["git", "-C", str(sub), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        # Submodule not initialized — fall back to the gitlink recorded in the superproject.
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO), "ls-tree", "HEAD", "standards"],
                capture_output=True, text=True, check=True,
            ).stdout.split()
            if len(out) >= 3 and out[1] == "commit":
                return out[2]
        except Exception:
            pass
    return ""


_TREE_CACHE: dict = {}


def methods_tree(sha: str | None = None) -> str:
    """Tree hash of methods/solver at <sha> (or current submodule HEAD).

    Comparing the methods *tree* (not the commit sha) means a Standards bump that
    only touches runner/ or web/ — not the numerical methods — does NOT mark
    existing solutions stale. A solution is stale only if the code that produces it
    actually changed.
    """
    key = sha or "HEAD"
    if key in _TREE_CACHE:
        return _TREE_CACHE[key]
    val = ""
    try:
        val = subprocess.run(
            ["git", "-C", str(REPO / "standards"), "rev-parse", f"{key}:methods/solver"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        val = ""
    _TREE_CACHE[key] = val
    return val


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  ! could not parse {path.relative_to(REPO)}: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Flag solutions built against a stale methods pin.")
    ap.add_argument("--strict", action="store_true", help="exit nonzero if anything is stale")
    args = ap.parse_args()

    pin = methods_pin()
    if not pin:
        print("stale: could not determine the standards submodule pin "
              "(run: git submodule update --init).", file=sys.stderr)
        return 2
    short = pin[:10]
    cur_tree = methods_tree(None)
    print(f"methods pin (standards submodule): {short}"
          + (f"  (methods tree {cur_tree[:10]})" if cur_tree else ""))

    def is_stale(sha: str) -> bool:
        if not sha:
            return True
        if not cur_tree:                 # can't read trees — fall back to commit-sha compare
            return sha != pin
        t = methods_tree(sha)
        return t == "" or t != cur_tree  # stale if the methods/solver code actually differs

    # Index pool versions and their staleness.
    stale_versions: dict[tuple[str, str], str] = {}  # (problem, version) -> sha
    n_pool = 0
    if POOL.exists():
        for meta in sorted(POOL.glob("*/v*/meta.json")):
            data = load_json(meta)
            if data is None:
                continue
            n_pool += 1
            sha = data.get("standards_methods_sha", "")
            key = (data.get("problem", meta.parents[1].name), data.get("version", meta.parent.name))
            if is_stale(sha):
                stale_versions[key] = sha

    # by-tex locks affected by a stale pin.
    affected = []  # (stem, reason)
    n_locks = 0
    if BYTEX.exists():
        for lock in sorted(BYTEX.glob("*/lock.json")):
            data = load_json(lock)
            if data is None:
                continue
            n_locks += 1
            stem = lock.parent.name
            reasons = []
            for s in data.get("solutions", []):
                key = (s.get("problem"), s.get("version"))
                if key in stale_versions:
                    reasons.append(f"{key[0]}/{key[1]} is stale")
                if s.get("standards_methods_sha") and is_stale(s["standards_methods_sha"]):
                    reasons.append(f"{key[0]}/{key[1]} lock sha {s['standards_methods_sha'][:10]} methods != pin")
            if reasons:
                affected.append((stem, "; ".join(sorted(set(reasons)))))

    # Report.
    print(f"checked {n_pool} pool version(s), {n_locks} by-tex lock(s); releases/ skipped.")
    for (problem, version), sha in sorted(stale_versions.items()):
        print(f"STALE     {problem}/{version}  sha {sha[:10] or '<none>'} != pin {short}")
    for stem, reason in affected:
        print(f"AFFECTED  by-tex/{stem}  ({reason})")

    if not stale_versions and not affected:
        print("clean: all solutions and figures match the current methods pin.")
        return 0

    print(f"\n{len(stale_versions)} stale version(s), {len(affected)} affected figure(s).")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
