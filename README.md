# GQ-Mono3D

A geometry-quality-aware monocular 3D object detection project built upon 3D-MOOD, featuring query quality prediction, quality-aware score calibration, and residual geometry refinement.

This repository is a cleaned research implementation prepared for reproducibility and technical portfolio presentation.

## Highlights

1. Adapted and reproduced the 3D-MOOD baseline on the SUN RGB-D 10-class setting.
2. **Geometry Quality Head (GQH)**: Predicts query-level geometric quality scores.
3. **Soft Geometry-Aware Query Scoring (Soft-GAQS)**: Calibrates inference scores using classification confidence and predicted geometric quality.
4. **Residual Geometry Refinement Head (RGRH)**: Predicts residual corrections for depth and object dimensions with quality-aware dynamic weights.
5. Completed training, evaluation, ablation studies, quality analysis, inference-speed measurement, and result visualization.
6. Improved 3D mAP@0.50 from 0.2350 to 0.2676 with nearly unchanged inference speed.

## Method Overview

### Geometry Quality Head

The Geometry Quality Head (GQH) predicts a scalar geometric quality score for each decoder query based on its feature representation. This quality score provides geometry-aware reliability information for candidate scoring and dynamic refinement, enabling the model to distinguish between relatively reliable and unreliable geometric predictions.

### Soft-GAQS

Soft Geometry-Aware Query Scoring (Soft-GAQS) calibrates detection scores during inference:

```
S_calib = (1 - alpha) * S_cls + alpha * Q_pred
```

- `alpha = 0.3`
- Applied during inference only
- Used for final candidate ranking and before NMS
- Does not directly modify classification training loss

### Residual Geometry Refinement Head

The Residual Geometry Refinement Head (RGRH) predicts residual corrections for depth and object dimensions:

- Predicts geometric residuals for depth and 3D bounding box dimensions
- Uses tanh activation to constrain residual range
- Applies quality-aware dynamic weights to adjust refinement strength
- Dynamic weight formula:

```
w_i = 1 + eta * ReLU(Q_i - tau)
```

Parameters:
- `eta = 0.3`
- `tau = 0.5`
- Residual update scale = 0.10

## Results

| Method | Checkpoint | 3D mAP@0.25 | 3D mAP@0.50 | FPS |
|---|---|---:|---:|---:|
| 3D-MOOD baseline | baseline | 0.5299 | 0.2350 | 4.50 |
| GQ-Mono3D | best_map50 | 0.5201 | 0.2676 | 4.49 |

Notes:
- Both GQ-Mono3D metrics (0.5201 and 0.2676) come from the same `best_map50` checkpoint
- mAP@0.50 improved by 0.0326 absolutely, approximately 13.9% relatively
- mAP@0.25 decreased by 0.0098
- Results from offline evaluation on SUN RGB-D test split (10 classes)
- Checkpoints and datasets are not included in this repository

Resources:
- [Evaluation summary](results/evaluation/summary.csv)
- [Sanitized best-checkpoint report](results/evaluation/best_map50_report.txt)
- [Evaluation details](results/README.md)

## Dataset and Evaluation

- **Dataset**: SUN RGB-D test split
- **Test images**: 5050
- **Classes**: bathtub, bed, bookshelf, chair, desk, dresser, nightstand, sofa, table, toilet
- **Evaluator**: Omni3DEvaluator
- **Metrics**: 3D mAP@0.25 and 3D mAP@0.50

## Repository Structure

- `opendet3d/`: Core implementation including model, operator, and evaluation modules
- `opendet3d/zoo/gdino3d/gdino3d_swin_t_sunrgbd10.py`: SUN RGB-D 10-class configuration
- `scripts/`: Demo and evaluation scripts
- `analyze_q_stats.py`: Geometric quality statistics analysis from inference logs
- `analyze_q_depth.py`: Correlation analysis between geometric quality and predicted depth
- `analyze_q_score_corr.py`: Correlation analysis between geometric quality and detection scores
- `results/`: Evaluation results and reports

## Installation

This project inherits dependencies from 3D-MOOD and Vis4D. Please refer to the original 3D-MOOD documentation for detailed environment setup.

```bash
git clone <your-repository-url>
cd GQ-Mono3D
pip install -r requirements.txt
```

The project follows the Vis4D-based environment of the upstream 3D-MOOD implementation. Please refer to the original 3D-MOOD setup instructions and check `requirements.txt` and `Dockerfile` for dependency information.

## Training and Evaluation

Training and evaluation entry points are provided through Vis4D CLI and scripts. Dataset paths and pretrained weights must be configured locally before execution.

Key scripts:
- `scripts/demo.py`: Official 3D-MOOD demo
- `scripts/demo_my.py`: Custom demo with geometric quality visualization
- `scripts/run_eval.py`: Evaluation script for Omni3D format datasets

The primary configuration file for SUN RGB-D 10-class experiments is located at `opendet3d/zoo/gdino3d/gdino3d_swin_t_sunrgbd10.py`.

## Analysis Scripts

- `analyze_q_stats.py`: Extracts and computes statistics for geometric quality scores from inference logs, including mean, standard deviation, min/max values, and percentile distributions.
- `analyze_q_depth.py`: Analyzes the correlation between predicted geometric quality and predicted depth, generating scatter plots and binned statistics.
- `analyze_q_score_corr.py`: Analyzes the correlation between predicted geometric quality and detection confidence scores, generating scatter plots and binned statistics.

## Acknowledgements and Attribution

This project is built upon the [3D-MOOD](https://github.com/cvg/3D-MOOD) codebase. We thank the original authors for their contributions.

The upstream 3D-MOOD code is distributed under the Apache License 2.0. This repository retains the upstream license and attribution. Modifications specific to GQ-Mono3D are described in this README.

3D-MOOD is described in:

> [**3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection**](https://royyang0714.github.io/3D-MOOD) \
> Yung-Hsu Yang, Luigi Piccinelli, Mattia Segu, Siyuan Li, Rui Huang, Yuqian Fu, Marc Pollefeys, Hermann Blum, Zuria Bauer \
> ICCV 2025

```bibtex
@InProceedings{Yang_2025_ICCV,
    author    = {Yang, Yung-Hsu and Piccinelli, Luigi and Segu, Mattia and Li, Siyuan and Huang, Rui and Fu, Yuqian and Pollefeys, Marc and Blum, Hermann and Bauer, Zuria},
    title     = {3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {7429-7439}
}
```

## Disclaimer

- This repository is intended for academic research and technical portfolio demonstration.
- Dataset files and model checkpoints are not included.
- Results are reported from offline evaluation.
- This is not production-ready software and is not designed for real vehicle deployment.