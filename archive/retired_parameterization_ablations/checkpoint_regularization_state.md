# Retired solve-regularization checkpoint state

The former bounded-alpha ImageNet recipe attached two RRLSSO-specific losses:
a log-gain reference anchor and an alpha saturation hinge. Supporting them
required stage-aware checkpoint state beyond ordinary optimizer resumption.

The retired checkpoint contract included:

- `rrlsso_gain_reference` tensors in `last.pt` and `best.pt`;
- `gain_reference_origin` in run metadata;
- separate reference initialization for new pretraining and new refinement;
- reference restoration without rebasing during a true stage resume;
- regularizer weights, saturation threshold, and extended gradient diagnostics;
- gate assertions that hashed the reference before and after resume.

This logic was removed when the two explicit regularizers and the alpha upper
bound were removed. The last complete implementation is preserved in Git
commit `3b53f47` (`experiments/imagenet_wds_train.py` and
`experiments/deit3_official_recipe.py`).

The active `init_finetune` versus `resume_finetune` state machine is **not**
part of this retired mechanism. It remains necessary:

- a new refinement loads model weights, resizes positional embeddings, and
  creates fresh optimizer, scheduler, scaler, and EMA state;
- a true resume restores the stage-local model, EMA, optimizer, scheduler,
  scaler, epoch/update counters, best score, and RNG state.

Keeping those operations distinct prevents an initialization checkpoint from
silently overwriting an in-progress refinement run.
