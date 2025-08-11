import torch

@torch.no_grad()
def collect_logits_feats(model, loader, device):
    model.eval()
    all_logits, all_feats, all_labels = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits, feats = model(xb, return_feat=True)
        all_logits.append(logits.cpu())
        all_feats.append(feats.cpu())
        all_labels.append(yb)
    return torch.cat(all_logits), torch.cat(all_feats), torch.cat(all_labels)

def head_logits_from_feats(model, feats):
    head_device = next(model.head.parameters()).device
    with torch.no_grad():
        return model.head(feats.to(head_device)).cpu()
