# SEMTRA-MML

**SEMTRA-MML: A Motion Semantics-Enhanced Multi-Modal Learning Framework for Trajectory Representation Learning** (IEEE TITS, under review).

SEMTRA-MML is a dual-stream hierarchical framework for trajectory representation learning. It jointly encodes **Road-GPS dynamics** and **Spatial Grid** streams, and injects human-interpretable motion semantics through **learnable motion anchors** that are bound to twelve rule-defined, verifiable motion concepts (direction, road context, and speed regime) via a supervised semantic-anchoring loss. A dual self-supervised objective (cross-modal alignment + semantic structure regularization) keeps the learned space consistent and plausibly structured.

![model](docs/fig2_framework.png)

## Model overview

- **Multi-modal embedding**: road-segment embeddings (GAT-pretrained), per-point GPS dynamics embeddings (longitude/latitude/heading, no time-derived kinematics), and CNN-based grid embeddings.
- **Dual-stream hierarchical encoder**: shallow encoder → motion semantics enhancement (anchor cross-attention) → deep encoder, for each stream.
- **Supervised semantic anchoring**: every anchor is tied to a rule-defined concept whose occurrences are verifiable on the map; a quality gate (linear-probe AUC ≥ 0.8) admits only reliable concepts.
- **Objectives**: `L_total = L_main + α (L_align + λ_struct · L_struct + λ_sem · L_sem)`.

## Repository layout

```
semtra_mml/
  __init__.py
  models/
    __init__.py
    semtra.py        # full model definition (SEMTRA_MML + all sub-modules)
example.py           # build the model, count parameters, run a dummy forward pass
```

This repository contains the **model code** of SEMTRA-MML in a single self-contained module (`semtra_mml/models/semtra.py`, PyTorch only). Training/data pipelines and third-party baseline implementations are not included.

## Quick start

```bash
pip install -r requirements.txt
python example.py
```

`example.py` instantiates SEMTRA-MML (Base configuration: hidden dim 128, 2 shallow + 4 deep Transformer layers, 12 motion anchors), prints the parameter count, and runs a dummy forward pass on synthetic inputs.

## Data

| Dataset | Size | Link |
|---|---|---|
| Porto | 843,920 trajectories | included in this repository (see `data/`) |
| Xiamen | 5,471 trajectories | included in this repository |
| Shanghai | 948,705 trajectories | [Zenodo DOI — to be added] |

All datasets follow the same preprocessed multi-modal format (road-ID sequences, grid-ID sequences, and GPS point features per trajectory).

## Citation

If you use this code, please cite:

```bibtex
@article{yin2026semtramml,
  author  = {Yin, Jie and Li, Donghuan and Hou, Yue and Lu, Keke},
  title   = {SEMTRA-MML: A Motion Semantics-Enhanced Multi-Modal Learning Framework for Trajectory Representation},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
