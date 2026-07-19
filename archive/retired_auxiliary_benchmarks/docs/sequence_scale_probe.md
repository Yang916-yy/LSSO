# Archived sequence benchmark scale probe (2026-07-13)

This is a pre-experiment engineering probe, not a paper result. All model-size
decisions below use validation accuracy only. Test metrics produced by the
runners were deliberately excluded from model selection.

## Protocol

- Mixer: RRLSSO.
- Seed: 0.
- Genomic: 12,000 training and 2,000 validation examples, 8 epochs,
  `human_enhancers_cohn` and `human_nontata_promoters`.
- UEA: the complete LSST training split with a fixed 90/10 stratified
  train/validation split, 50 epochs, rank-16 learned position factorization.
- LRA: byte-level IMDB Text, length 4096, 4,000 training and 1,000 validation
  examples, 15 epochs.
- Throughput and peak memory are local measurements and are useful only for
  relative comparisons on this machine.

## Results

### GenomicBenchmarks

| Task | Width x depth | Rank | Parameters | Best validation | Samples/s | Peak GB |
|---|---:|---:|---:|---:|---:|---:|
| Enhancers Cohn | 64 x 2 | 8 | 125,074 | 67.95% | 11,666 | 0.797 |
| Enhancers Cohn | 128 x 2 | 16 | 430,354 | 67.90% | 6,619 | 1.592 |
| Enhancers Cohn | 128 x 4 | 16 | 777,378 | 67.85% | 3,597 | 2.900 |
| Enhancers Cohn | 192 x 4 | 32 | 1,768,562 | 68.20% | 1,820 | 4.552 |
| Non-TATA promoters | 64 x 2 | 8 | 125,074 | 73.85% | 17,638 | 0.413 |
| Non-TATA promoters | 128 x 2 | 16 | 430,354 | 75.45% | 12,539 | 0.838 |
| Non-TATA promoters | 192 x 2 | 32 | 952,922 | 76.30% | 6,652 | 1.281 |

Increasing the 128 x 4 learning rate from `3e-4` to `1e-3` on Cohn did not
recover a hidden large-model advantage (67.85% versus 67.70%). The provisional
common preset is therefore **128 x 2, rank 16**. The 192 x 2 configuration is
the accuracy-oriented upper bracket for the second-stage sweep.

### UEA

| Width x depth | Rank | Parameters | Best validation | Samples/s | Peak GB |
|---:|---:|---:|---:|---:|---:|
| 32 x 2 | 8 | 26,174 | 48.57% | 12,661 | 0.033 |
| 64 x 3 | 8 | 138,438 | 57.96% | 9,689 | 0.061 |
| 128 x 4 | 16 | 716,014 | 66.12% | 7,853 | 0.136 |
| 192 x 4 | 32 | 1,676,222 | 69.80% | 7,462 | 0.215 |
| 256 x 4 | 32 | 2,972,046 | 71.02% | 7,083 | 0.285 |

The provisional common preset is **192 x 4, rank 32, position rank 16**. Going
to width 256 adds only 1.22 points while increasing parameters by 77%, so it is
kept as an upper bracket rather than the default. This choice must still be
checked on at least one very small UEA dataset because LSST is relatively large.

### LRA Text

| Width x depth | Rank | Parameters | Best validation | Samples/s | Peak GB |
|---:|---:|---:|---:|---:|---:|
| 64 x 2 | 8 | 362,706 | 63.50% | 2,436 | 0.844 |
| 128 x 2 | 16 | 905,618 | 63.50% | 1,930 | 1.622 |
| 128 x 4 | 16 | 1,252,642 | 62.80% | 1,107 | 2.949 |
| 256 x 1 | 16 | 1,841,042 | 64.00% | 2,042 | 1.990 |
| 256 x 1 | 32 | 1,873,938 | 64.30% | 1,713 | 2.433 |
| 256 x 2 | 32 | 2,597,922 | 63.50% | 951 | 3.365 |

The corrected, stratified runs use the same limited-data protocol. Width 256
was more useful than increasing depth from two to four at width 128. Although
the one-layer pilot reached the highest Text validation score, the formal
cross-task preset is deliberately frozen at **256 x 2, rank 32**: two blocks
provide a less task-specific representation stack and a cleaner architecture
for the paper. The one-layer result is retained as a scale ablation rather than
used to tune the formal model to this pilot. A ListOps check and a three-seed
confirmation remain required.

## Launch decision

Do not choose a different architecture after looking at every individual test
set. Use one preset per benchmark suite, freeze it using validation-only pilot
runs, and then run all mixers and seeds under that preset. The next scale gate
is a three-seed confirmation on representative tasks:

- Genomic: Cohn and Non-TATA, widths 128 and 192 at depth 2.
- UEA: one small dataset plus LSST, widths 192 and 256 at depth 4.
- LRA: Text plus ListOps, 128 x 2 versus the frozen 256 x 2 formal preset;
  retain 256 x 1 and rank 16 only as scale ablations.

Only if the upper bracket improves the validation result should it replace the
provisional preset. Genomic architecture exploration uses seed 0 only; repeated
seeds are reserved for the frozen formal configuration. This avoids spending
the formal seed budget during model selection and prevents silently tuning
model size on benchmark test sets.
