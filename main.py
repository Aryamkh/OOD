from oodx.runner import run
import argparse
from pathlib import Path

def build_argparser():
    p = argparse.ArgumentParser(description="Train ID model and evaluate OOD detectors")
    # Data / general
    p.add_argument("--id_dataset",  type=str, default="warcoder/soyabean-seeds")
    p.add_argument("--ood_dataset", type=str, default="neelgajare/rocks-dataset")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--workers",     type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)

    # Backbone / input
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18","resnet50","convnext_tiny"])
    p.add_argument("--img_size", type=int, default=128)

    # Head options
    p.add_argument("--cosine_head", type=int, default=0, help="1 to use cosine classifier")
    p.add_argument("--cosine_scale", type=float, default=30.0, help="scale for cosine head logits")

    # Train regularizers
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0, help="alpha; 0 disables")
    p.add_argument("--cutmix", type=float, default=0.0, help="alpha; 0 disables")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay; 0 disables")
    p.add_argument("--warmup_epochs", type=int, default=0)

    # ODIN (pre-norm FGSM)
    p.add_argument("--enable_odin", type=int, default=0)
    p.add_argument("--odin_eps", type=float, default=0.002)  # in [0,1] pre-norm pixel space
    p.add_argument("--odin_temp", type=float, default=1000.0)

    # Post-hoc toggles
    p.add_argument("--enable_react", type=int, default=0)
    p.add_argument("--react_percent", type=float, default=0.9)  # clamp at p-th percentile

    p.add_argument("--enable_ash", type=int, default=0)
    p.add_argument("--ash_p", type=float, default=0.9)  # drop top-p activations

    p.add_argument("--enable_vim", type=int, default=0)
    p.add_argument("--vim_alpha", type=float, default=0.05)

    p.add_argument("--enable_adascale", type=int, default=0)
    p.add_argument("--adascale_gamma", type=float, default=0.2)  # lightweight heuristic

    # OOD training/eval options
    p.add_argument("--use_oe",      type=int, default=0, help="1 to enable Outlier Exposure during training")
    p.add_argument("--oe_lambda",   type=float, default=0.5)

    p.add_argument("--outdir",      type=str, default="results")
    return p

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    if args.backbone == "convnext_tiny" and args.img_size < 224:
        print("Auto-setting --img_size 224 for ConvNeXt-Tiny")
        args.img_size = 224
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    run(args)
