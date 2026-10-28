# PSTS and RCCR: Core Source Code

Core source files for the dual-decoder medical image segmentation implementation.
The files retain their original contents and internal directory structure.

## File Map

- `code/train_Mutual_Reliable.py`: training, routing, and reliability weighting.
- `code/test_3d.py`: three-dimensional inference and evaluation.
- `code/networks/VNet.py`: V-Net-based network architectures.
- `code/networks/net_factory.py`: network construction.
- `code/networks/unet2d.py`: dependency imported by the network factory.
- `code/dataloaders/dataset.py`: datasets, transforms, and batch sampling.
- `code/utils/rcts_module.py`: perturbation-stability routing (`ps_wr`) and related routing utilities.
- `code/utils/hr_module.py`: prototype-based reliability modeling.
- `code/utils/rccr_module.py`: cross-view agreement, regional calibration, and selective suppression.
- `code/utils/rccr_diagnosis.py`: recorder imported by the training entry point.
- `code/utils/losses.py`, `metrics.py`, `ramps.py`, and `test_patch.py`: required training and evaluation utilities.

## Scope

This is a source-only package, not a validated reproduction release. It contains
no datasets, experiment outputs, checkpoints, prediction masks, logs, caches,
editor settings, or obsolete source backups. Original algorithms, default
parameters, and local path defaults have not been changed. The local Spleen
training-sample default remains 33; this snapshot has not been reconciled with
the manuscript's reported 30-case training split. Dataset paths, exact final
experiment configurations, and the correspondence to submitted results require
author confirmation. No training or reproduction was performed during packaging.

## Upstream Acknowledgment

This implementation builds on **Mutual learning with reliable pseudo label for
semi-supervised medical image segmentation**:

- Paper: https://doi.org/10.1016/j.media.2024.103111
- Original repository: https://github.com/Jiawei0o0/mutual-learning-with-reliable-pseudo-labels

The original project also acknowledges code adapted from
[MC-Net](https://github.com/ycwu1997/MC-Net) and
[SSL4MIS](https://github.com/HiLab-git/SSL4MIS).
Existing source notices are retained. This packaging step does not assign a new
license or establish permission to relicense upstream code.
