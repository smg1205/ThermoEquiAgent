# Training protocol

## Stage 1: supervised optimization

Stage 1 optimizes pressure, temperature, vapor composition, and the supervised
pure-component endpoint vapor-pressure term. Group-disjoint validation selects
the checkpoint. The pure endpoint term is a data target, not an additional
Stage-2 physical constraint.

## Stage 2: fugacity-constrained fine-tuning

Stage 2 loads the best Stage-1 validation checkpoint and minimizes

```text
L_total = L_supervised + lambda_fugacity L_teacher_forced_fugacity.
```

The fugacity residual uses observed `T`, `P`, `x`, and `y`:

```text
x_i gamma_i P_i^sat(T) - y_i P.
```

It is therefore named `teacher_forced_fugacity` and is not presented as a free
solver residual. The additional loss is warmed up over two epochs. Fine-tuning
runs for 10 epochs and updates only:

| Parameter group | Learning rate |
|---|---:|
| `pair_potential` | `2e-5` |
| `vapor_pressure` | `1e-5` |
| `film` | `5e-6` |
| `mixture_token` | `5e-6` |

The Stage-1 checkpoint is included as epoch zero. Validation may retain Stage 1
when Stage 2 reduces predictive performance. The test partition is evaluated
once after checkpoint selection.
