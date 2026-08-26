
# Method Overview

## Final submitted system

The final submission is a **final single-model student** for nine ultrasound
biometry tasks. Runtime uses one model graph, one checkpoint, and one image
view. Five fused teachers are used only offline to generate
soft targets over the official unlabeled pool. Eight task routes use
reliability-weighted unlabeled distillation; Femur retains its supervised
route within the same nine-task student.

The organizer-provided task identifier selects a task-specific heatmap route.
A shared ConvNeXt-Small pathway provides local landmark evidence, and a USFM
BEiT-B/16 pathway provides complementary ultrasound-domain representations.
Feature and heatmap evidence is aligned at 64 x 64 resolution and combined by
reference-preserving dense residual fusion.

## Stable landmark identity

A4C and PSAX use canonical endpoint semantics during supervised fitting,
checkpoint evaluation, teacher aggregation, and student distillation. A single
deterministic conversion to the organizer's vertical endpoint order is applied
at the output boundary. This keeps channel meaning fixed while preserving the
official output contract.

## Task-specific configuration

| Task | Points | Task-specific design | Unlabeled adaptation |
| --- | ---: | --- | --- |
| A4C | 16 | Stable paired endpoint identity | Reliability-weighted distillation |
| AOP | 4 | Shared angle vertex and HSD geometry | Reliability-weighted distillation |
| FA | 4 | Task-specific dense heatmap head | Reliability-weighted distillation |
| FUGC | 2 | Task-specific dense heatmap head | Reliability-weighted distillation |
| HC | 4 | Task-specific dense heatmap head | Reliability-weighted distillation |
| IVC | 2 | Task-specific dense heatmap head | Reliability-weighted distillation |
| PLAX | 22 | Task-specific dense heatmap head | Reliability-weighted distillation |
| PSAX | 4 | Stable identity plus endpoint, midpoint, tube, and direction supervision | Reliability-weighted distillation |
| Femur (fetal femur) | 2 | Task-specific dense heatmap head | Supervised route retained |

## Deployment lock

- CodaBench submission ID: `879954`
- Docker image digest: `sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e`
- Deployment checkpoint SHA256: `73086754757a70be87319d636ab4b837aeeefbed88e2189add40484cfb7b8da1`
- Official validation: Average MRE `24.798556`, Average MAE `25.923333`
- Hidden test: Average MRE `24.9356305937`, Average MAE `26.8757741711`

Leaderboard A / Method 1: preliminary rank 12/31, score 0.28363.
Leaderboard B / Method 2: preliminary rank 15/31, score 0.27864.
The organizer-reported PSAX MRE is 39.311, preliminary rank 1 in the PSAX MRE metric dimension. Ranking
statements remain subject to organizer eligibility review.

Public method repository: https://github.com/cjunxiao/FoundUS2026
