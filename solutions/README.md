# solutions

Immutable solution versions and the figure→version pins.

```
solutions/
├── REGISTRY.json              index of all pool versions (problem, version, sha, metrics)
├── _templates/
│   ├── meta.json              canonical schema for a version's meta.json
│   └── lock.json              canonical schema for a by-tex lock.json
├── pool/<problem>/vNNNN/      immutable solution version
│   ├── data/                  array(s) as compressed binary .npz
│   ├── figure/                ready PDF(s) produced from this version
│   └── meta.json              params, metrics, standards_methods_sha
└── by-tex/<tex-stem>/         per-figure binding
    ├── lock.json              pins the exact pool version(s) this tex uses
    └── figures/               copies of the small ready PDFs the tex includes
```

## Rules

- **Immutable + monotonic.** `vNNNN` is allocated once and never overwritten. A new
  run is a new version.
- **Binary `.npz`.** Arrays are stored with `np.savez_compressed` — never 4-byte
  string arrays. If a single artifact exceeds ~25 MB, move that version's `data/`
  to Git LFS or an external store and note it in `meta.json`.
- **Provenance.** Every `meta.json` records `standards_methods_sha`: the Standards
  submodule commit whose methods produced it. `make stale` flags any version whose
  sha ≠ the current pin.
- **by-tex pins, doesn't duplicate.** A figure's `lock.json` references heavy pool
  data by (problem, version); only the small ready PDFs are copied into
  `by-tex/<stem>/figures/`.

`REGISTRY.json` is the lightweight index; it is appended to whenever a new version
is written (see `numerics/<problem>/solve.py`).
