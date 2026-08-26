
# Final Distillation

## Scope

The submitted system uses reliability-weighted unlabeled ensemble
distillation. This is the final unlabeled-learning pathway represented by the
Docker artifact. The controlled appearance-replay study is independent and is
not part of the submitted runtime graph.

## Teacher targets

Five fixed fold-specific fused predictors process each official unlabeled
image. The mean per-landmark heatmap is retained as the spatial target.
Coordinate dispersion across teachers is converted into a continuous
reliability weight, so uncertain targets contribute less without converting
them into hard pseudo-coordinates.

All 182,870 readable, deduplicated official unlabeled images across eight
task routes receive teacher targets. Of these, 181,846 unique images participate in optimization and 1,024
are held out for label-free audit.

## Student update

The student starts from a preselected fold-1 medoid fused predictor fixed before
hidden-test evaluation. ConvNeXt, original task heads, fusion layers, and base
USFM parameters remain frozen. Training updates task-specific rank-4 LoRA
modules in the final four USFM Transformer blocks and zero-initialized
contextual heatmap corrections. The final stage adapts 969,852 parameters.

The objective is:

```text
L = <w, KL(P_teacher || P_student)>
    + 0.05 <w, SmoothL1(c_teacher, c_student)>
    + L_reg
```

Reliability weights therefore apply to both heatmap KL divergence and decoded
coordinate consistency. Femur has no official unlabeled pool and retains
its supervised route.
