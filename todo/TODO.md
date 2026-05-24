# TODO

## Done
- [x] Scaffold MIWN + pin Standards as the `standards/` submodule.
- [x] Wire `numerics/ree_K3/solve.py` to the shared methods (symmetric K solver).
- [x] T6 cross-repo runner — **live**: Standards `runner/run_task.py` +
      `claim_task` `QUEUE_REL`; MIWN `.github/workflows/solve.yml` (2 Claude
      workers); submodule bumped to the merged commit.
- [x] Migrate curated REZN content (manuscript → `tex/paper.tex`; 5 converged
      solutions → `solutions/pool/ree_K3/v0003–v0007`; 15 figures → `by-tex/`;
      proofs/notes → `brainstorm/rezn/`).
- [x] Figure pipeline: 77-point sweep `todo/TASK_QUEUE.json`, `tex/FIGURES.md`,
      no-learning `solutions/figures_data.json`.
- [x] Website (runner dashboard) in Standards `runner/web/`, pointed at MIWN, with
      a Pages deploy workflow.
- [x] Robustness: `solve.py` wall-cap + branch guard; runner skip-on-reject
      (workers can't hang or commit wrong-branch data).

## Open
- [ ] **Owner toggle:** enable Pages on Standards (Settings → Pages → Source =
      "GitHub Actions") so the dashboard serves at
      `https://mhpbreugem.github.io/Standards/`.
- [ ] **Continuation driver (blocks the sweep):** make `solve.py` warm-start from
      the task's `depends_on` checkpoint instead of cold-starting, so each step
      stays on the strong-PR branch radiating from the `v0007` seed. The spike
      showed cold starts collapse to the fully-revealing root (1−R²≈3e-5 vs ref
      ≈0.085); until this lands, workers correctly **skip** sweep tasks.
- [ ] Run the full (γ, τ) sweep via the runner once the driver lands; populate
      `solutions/pool/` and regenerate `figures_data.json` (REE lines).
- [ ] Confirm the best-effort figure→solution pins in `solutions/by-tex/*/lock.json`
      against each figure's generating script (currently all `v0007`, except
      `fig5_convergence` → G-ladder v0003/v0005/v0006).
- [ ] Update Standards `methods/MAP.md` provenance to the archived REZN commit
      (Standards PR); then archive the REZN repo (owner, GitHub Settings).
- [ ] Wire `make publish` end-to-end once `tex/paper.tex` compiles with real figures.
