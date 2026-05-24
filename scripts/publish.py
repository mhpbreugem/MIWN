#!/usr/bin/env python3
"""
publish.py — assemble a release snapshot (make publish VERSION=<label>).

Light part (committed) -> releases/<label>/: paper.pdf, MANIFEST.json, README.md.
Heavy part (NOT committed) -> miwn-<label>.zip: frozen tex/, only-used pool
results, a vendored copy of standards/methods at the submodule commit,
requirements.txt, plus the manifest + reproduce README.

Prints the `gh release create` command for the heavy bundle; never runs it, never
git-adds the zip. releases/ is excluded from `make stale`.

  python3 scripts/publish.py --version ecta-accepted
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BYTEX = REPO / "solutions" / "by-tex"
POOL = REPO / "solutions" / "pool"
STANDARDS = REPO / "standards"


def git(args: list[str], cwd: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def resolve_usage() -> tuple[list[dict], list[Path]]:
    """From every by-tex lock.json: the (problem, version) set + figure files used."""
    used: dict[tuple[str, str], dict] = {}
    figures: list[Path] = []
    if not BYTEX.exists():
        return [], []
    for lock in sorted(BYTEX.glob("*/lock.json")):
        data = json.loads(lock.read_text())
        for s in data.get("solutions", []):
            used[(s["problem"], s["version"])] = {"problem": s["problem"], "version": s["version"]}
        for fig in data.get("figures", []):
            p = lock.parent / fig
            if p.exists():
                figures.append(p)
    return list(used.values()), figures


def compile_paper(out_pdf: Path) -> bool:
    """Compile tex/paper.tex -> out_pdf. Returns True on success."""
    engine = shutil.which("latexmk") or shutil.which("pdflatex")
    if not engine:
        print("publish: WARNING — no LaTeX engine (latexmk/pdflatex) found; "
              "skipping paper.pdf. Install TeX Live to include it.", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as td:
        if engine.endswith("latexmk"):
            cmd = [engine, "-pdf", "-interaction=nonstopmode", f"-outdir={td}", "paper.tex"]
        else:
            cmd = [engine, "-interaction=nonstopmode", f"-output-directory={td}", "paper.tex"]
        r = subprocess.run(cmd, cwd=str(REPO / "tex"), capture_output=True, text=True)
        built = Path(td) / "paper.pdf"
        if r.returncode != 0 or not built.exists():
            print(f"publish: WARNING — LaTeX compile failed (rc={r.returncode}); "
                  "skipping paper.pdf.", file=sys.stderr)
            return False
        shutil.copy2(built, out_pdf)
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble a MIWN release snapshot.")
    ap.add_argument("--version", required=True, help="release label, e.g. ecta-accepted")
    args = ap.parse_args()
    label = args.version

    if not STANDARDS.exists():
        print("publish: standards submodule not initialized "
              "(git submodule update --init).", file=sys.stderr)
        return 2
    # Preconditions (warn, don't hard-fail, so a dev can dry-run).
    if git(["status", "--porcelain"]):
        print("publish: WARNING — working tree is not clean; snapshot reflects "
              "your working copy, not a committed state.", file=sys.stderr)

    standards_sha = git(["rev-parse", "HEAD"], cwd=STANDARDS)
    try:
        miwn_sha = git(["rev-parse", "HEAD"])
    except Exception:
        miwn_sha = "uncommitted"

    used, figures = resolve_usage()
    print(f"publish {label}: {len(used)} solution version(s) used, {len(figures)} figure(s).")

    rel_dir = REPO / "releases" / label
    rel_dir.mkdir(parents=True, exist_ok=True)

    # --- paper.pdf (light) ---
    pdf_ok = compile_paper(rel_dir / "paper.pdf")

    manifest = {
        "release": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "miwn_commit": miwn_sha,
        "standards_commit": standards_sha,
        "paper_pdf": "paper.pdf" if pdf_ok else None,
        "solutions_used": used,
        "bundle_asset": f"miwn-{label}.zip",
        "doi": None,
        "build": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "requirements": "standards/methods/requirements.txt",
            "commands": [f"make publish VERSION={label}"],
        },
    }
    (rel_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    reproduce = f"""# MIWN release: {label}

Frozen snapshot. To reproduce:

```sh
unzip miwn-{label}.zip && cd miwn-{label}
pip install -r requirements.txt
# methods are vendored under code/standards-methods/ at standards commit {standards_sha[:10]}
```

- MIWN commit: `{miwn_sha}`
- Standards commit: `{standards_sha}`
- Solutions used: {", ".join(f"{u['problem']}/{u['version']}" for u in used) or "(none)"}

The heavy bundle (frozen tex/, used results, vendored methods) is the GitHub
Release asset `miwn-{label}.zip`. This `releases/{label}/` folder holds only the
light in-git snapshot (paper.pdf + MANIFEST.json + this README).
"""
    (rel_dir / "README.md").write_text(reproduce)
    print(f"publish: wrote light snapshot -> releases/{label}/")

    # --- heavy bundle (NOT committed) ---
    zip_path = REPO / f"miwn-{label}.zip"
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / f"miwn-{label}"
        stage.mkdir()
        # frozen tex
        shutil.copytree(REPO / "tex", stage / "tex")
        # only-used pool results
        for u in used:
            src = POOL / u["problem"] / u["version"]
            if src.exists():
                dst = stage / "results" / u["problem"] / u["version"]
                shutil.copytree(src, dst)
        # vendored methods at the submodule commit (frozen)
        (stage / "code").mkdir()
        archive = subprocess.run(
            ["git", "-C", str(STANDARDS), "archive", "--format=tar",
             f"--prefix=standards-methods/", standards_sha, "methods"],
            capture_output=True, check=True,
        ).stdout
        tar_path = stage / "code" / "methods.tar"
        tar_path.write_bytes(archive)
        shutil.unpack_archive(str(tar_path), str(stage / "code"), format="tar")
        tar_path.unlink()
        # requirements + manifest + readme
        req = STANDARDS / "methods" / "requirements.txt"
        if req.exists():
            shutil.copy2(req, stage / "requirements.txt")
        shutil.copy2(rel_dir / "MANIFEST.json", stage / "MANIFEST.json")
        shutil.copy2(rel_dir / "README.md", stage / "README.md")
        if pdf_ok:
            shutil.copy2(rel_dir / "paper.pdf", stage / "paper.pdf")
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=td, base_dir=f"miwn-{label}")
    size_mb = zip_path.stat().st_size / 1e6
    print(f"publish: wrote heavy bundle -> {zip_path.name} ({size_mb:.1f} MB, NOT committed)")

    print("\nNext (run manually — do not commit the zip):")
    print(f"  gh release create {label} --title \"MIWN {label}\" "
          f"--notes-file releases/{label}/README.md  {zip_path.name}")
    print("  # then connect Zenodo to the repo to mint a DOI from the Release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
