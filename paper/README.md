# LSSO paper draft

This directory contains the theory-first front half of the LSSO/RRLSSO paper.
It intentionally stops after the implementation and complexity section; the
experimental protocol, results, limitations, and conclusion will be added once
the current training runs are complete.

Build from Windows with the installed MiKTeX toolchain:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The compiled draft is `build/main.pdf`.
