# Reproducibility protocol

## Frozen matrix

All formal experiments use bits 16/32/64/128 and seeds 2096/6513/9231.

| Family | Conditions | Runs |
|---|---:|---:|
| Full | 3 datasets x 4 bits x 3 seeds | 36 |
| Structural ablation | 4 conditions x 3 datasets x 4 bits x 3 seeds | 144 |
| Static-S1 | 3 datasets x 4 bits x 3 seeds | 36 |
| PairBlind Oracle | 3 datasets x 4 bits x 3 seeds | 36 |
| Sensitivity | frozen 64-bit validation grid | 45 |
| Non-CLIP robustness | MSCOCO, 64 bit, 3 conditions x 3 seeds | 9 |

The four structural ablations are No-Topology, BCE-Topology, No-Cross, and
No-Intra. Static-S1 updates model parameters only at stage 1 and still appends
codes for later arrivals.

## Evaluation

The retrieval gallery is append-only. Once a sample is committed, its binary
code is never recomputed. A query is relevant to a gallery item when their
multi-label vectors have a non-empty intersection. Both Img2Txt and Txt2Img are
reported at every stage, together with the historical-prefix gallery score.

The bundled evidence uses three-seed means and sample standard deviations for
formal statistics. Seed 6513 is used for qualitative curves and retrieval cases
only; it does not replace the three-seed analysis.

`evidence/runs.csv` is the sanitized provenance layer beneath these aggregates.
It contains 342 logical records spanning Full, UIH, structural ablations,
Static-S1, PairBlind, sensitivity, and non-CLIP robustness. Each row includes
the original artifact SHA-256 and a canonical row SHA-256. Sensitivity contains
45 preregistered logical points backed by 39 physical runs because the default
setting is shared across the three parameter sweeps.

## Recovery and aggregation

Every run writes its effective config, protocol identity, metrics, training log,
and completion status. Use `--resume` to skip a matching completed configuration.
Use `--retry-failed` only for a matching failed configuration. Aggregation reads
completed manifests and never selects hyperparameters from test results.

The IAPR schedule was selected using validation data only and was then frozen
for all code lengths and structural ablations.
