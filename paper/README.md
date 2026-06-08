# LSSO Paper Draft

`main.tex` is a compact six-page preprint-style draft built from the completed
experiments under `paper_results/`. It contains:

- the LSSO formulation and Woodbury implementation;
- fixed-U well-posedness, spectral, norm, conditioning, and energy results;
- retrieval, transfer, ablation, diagnostics, sequence-scaling, and vision
  results;
- explicit scope and limitations;
- a full Woodbury derivation and implementation notes in the appendix.

Compile with the bundled Tectonic binary or a normal LaTeX installation:

```powershell
tectonic main.tex --outdir build
```

The checked draft is available as `LSSO_draft.pdf`; local compile intermediates
are written to `build/` and are not intended for version control.

Before submission, replace `Anonymous Authors`, choose the target venue style,
add dataset and training-detail citations, and complete the missing formal
experiments noted in the repository roadmap.
