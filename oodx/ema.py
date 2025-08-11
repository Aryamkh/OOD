from copy import deepcopy
import torch

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p, alpha=1 - d)

    def copy_to(self, model):
        model.load_state_dict(self.ema.state_dict(), strict=False)
