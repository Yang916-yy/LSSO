# Retired CUDA token-RMS preparation path

The active LSSO implementation uses sample/head Gram-trace normalization.
The portable formula is preserved in
`archive/retired_parameterization_ablations/token_rms_reference.py`; it is no
longer exposed by supported constructors.

The native path retired here consisted of four custom operators:

- `normalize_basis`
- `normalize_basis_backward`
- `normalize_rank_rotary`
- `normalize_rank_rotary_backward`

They materialized a normalized basis and saved one FP32 inverse-RMS value per
token. This conflicts with the maintained statistics-first design, where the
normalization energy comes from `trace(U.T @ U)` without an extra pass or an
output-sized normalized-basis buffer.

The removed implementation remains recoverable from Git history. This
directory records the retirement boundary and is never compiled or imported.
