# tex

The manuscript source.

```
tex/
├── paper.tex            the manuscript (one paper.tex + per-figure .tex)
└── _provenance/
    └── lock.tex         \input-able record of which solution versions the figures pin
```

## Figures: pin via by-tex

Each figure has a `solutions/by-tex/<stem>/` folder. The small ready PDF is copied
into `solutions/by-tex/<stem>/figures/`, and `paper.tex` includes it from there —
never from the heavy pool. `solutions/by-tex/<stem>/lock.json` records the exact
pool version(s) the figure was built from; `_provenance/lock.tex` mirrors those
pins into the build so the compiled PDF is self-documenting.

Writing standards (pgfplots BC20→ECTA style, captions, notation) live in
`standards/writing/` — consult the relevant chapter before editing a figure or
section.

## Build

```sh
cd tex && pdflatex paper.tex     # or latexmk -pdf paper.tex
```

`make publish VERSION=<label>` compiles `paper.tex` → `paper.pdf` and assembles the
release snapshot (see `releases/README.md`).
