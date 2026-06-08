# LSSO Paper

`main.tex` is a compact seven-page preprint-style paper built from the completed
experiments under `paper_results/`. It contains:

- the LSSO formulation and Woodbury implementation;
- fixed-U well-posedness, spectral, norm, conditioning, and energy results;
- retrieval, transfer, ablation, diagnostics, sequence-scaling, and vision
  results;
- explicit scope and limitations;
- a compact Woodbury derivation in the appendix.

Compile with the bundled Tectonic binary or a normal LaTeX installation:

```powershell
tectonic main.tex --outdir build
```

The versioned paper is available as
`LSSO_Learnable_Low-Rank_Sylvester_Solves_for_Efficient_Bidirectional_Encoders_v1.pdf`;
local compile intermediates are written to `build/` and are not intended for
version control.

The current version lists Zhaoyang Wang as the author and links directly to the
repository from the title block.
