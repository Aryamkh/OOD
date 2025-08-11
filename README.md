# oodx — ID Training + OOD Evaluation

This repo splits your monolithic script into a small, importable package. It:
- downloads datasets from Kaggle via `kagglehub`
- trains an ID classifier (ResNet18/50 or ConvNeXt-Tiny)
- evaluates multiple OOD detectors (MSP, Energy, MaxLogit, Mahalanobis, kNN, ODIN*, ReAct*, ASH*, ViM*, AdaSCALE*)
- saves ROC/PR/hist plots and a 2D feature visualization

\* optional toggles; see CLI.

## Install

```bash
pip install -r requirements.txt
```

> Torch/Torchvision wheels depend on your CUDA/CPU. If you need specific versions, install them first per PyTorch docs.

## Run

```bash
python main.py   --id_dataset warcoder/soyabean-seeds   --ood_dataset neelgajare/rocks-dataset   --backbone resnet18   --epochs 50   --batch_size 64   --outdir results
```

Optional toggles (set to 1 to enable): `--enable_odin`, `--enable_react`, `--enable_ash`, `--enable_vim`, `--enable_adascale`, `--use_oe`.

All outputs (sample grids, ROC/PR/hist, feature plot) land in `results/` by default.

## Project layout

```
oodx/
  oodx/
    __init__.py
    utils.py
    data.py
    models.py
    ema.py
    augment.py
    train.py
    eval_utils.py
    scoring.py
    metrics_viz.py
    runner.py
  main.py
  requirements.txt
```

## Notes

- If you choose `--backbone convnext_tiny`, the script auto-sets `--img_size 224`.
- ODIN in this repo perturbs *pre-normalized* pixels for correctness when using normalized inputs.
