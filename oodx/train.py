import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from .augment import apply_mixup, apply_cutmix
from .ema import ModelEMA

def criterion_with_soft_targets(logits, target, label_smoothing=0.0, num_classes=None):
    if isinstance(target, tuple):
        y_a, y_b, lam = target
        return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + \
               (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
    else:
        return F.cross_entropy(logits, target, label_smoothing=label_smoothing)

def oe_uniform_loss(logits):
    return -F.log_softmax(logits, dim=1).mean()

def train(model, id_loader, ood_loader, epochs, lr, use_oe, oe_lambda, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    total_steps = epochs * max(1, len(id_loader))
    warmup_steps = max(0, args.warmup_epochs * len(id_loader))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ema = ModelEMA(model, decay=args.ema) if args.ema > 0 else None

    best_acc, best_state = 0.0, None
    model.train()
    step = 0
    for ep in range(1, epochs+1):
        losses, accs = [], []
        iter_ood = iter(ood_loader) if use_oe else None
        for xb, yb in id_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            target = yb
            if args.mixup > 0 and args.cutmix > 0:
                if np.random.rand() < 0.5:
                    xb, (ya, yb2), lam = apply_mixup(xb, yb, args.mixup)
                else:
                    xb, (ya, yb2), lam = apply_cutmix(xb, yb, args.cutmix)
                target = (ya, yb2, lam)
            elif args.mixup > 0:
                xb, (ya, yb2), lam = apply_mixup(xb, yb, args.mixup)
                target = (ya, yb2, lam)
            elif args.cutmix > 0:
                xb, (ya, yb2), lam = apply_cutmix(xb, yb, args.cutmix)
                target = (ya, yb2, lam)

            logits_id = model(xb)
            loss = criterion_with_soft_targets(
                logits_id, target,
                label_smoothing=args.label_smoothing,
                num_classes=logits_id.size(1)
            )

            if use_oe:
                try:
                    xo, _ = next(iter_ood)
                except StopIteration:
                    iter_ood = iter(ood_loader); xo, _ = next(iter_ood)
                xo = xo.to(device, non_blocking=True)
                logits_ood = model(xo)
                loss = loss + oe_lambda * (-F.log_softmax(logits_ood, dim=1).mean())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ema: ema.update(model)

            losses.append(loss.item())
            accs.append((logits_id.argmax(1) == yb).float().mean().item())

            step += 1

        mloss, macc = float(np.mean(losses)), float(np.mean(accs))
        if macc > best_acc:
            best_acc, best_state = macc, {k: v.detach().cpu() for k, v in (ema.ema if ema else model).state_dict().items()}
        print(f"Epoch {ep:03d}/{epochs} | loss {mloss:.4f} | acc {macc:.4f}")

    if best_state is not None:
        (ema.ema if ema else model).load_state_dict(best_state)
        if ema: ema.copy_to(model)
    return model
