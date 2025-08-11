from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch, numpy as np

def get_first_subdir(path):
    subdirs = [d for d in Path(path).iterdir() if d.is_dir()]
    return str(subdirs[0]) if len(subdirs) == 1 else path

def build_loaders_from_kaggle(id_dataset_name, ood_dataset_name, img_size=128, batch_size=64, workers=4):
    import kagglehub
    id_raw_path  = kagglehub.dataset_download(id_dataset_name)
    ood_raw_path = kagglehub.dataset_download(ood_dataset_name)
    print("Raw ID data at:", id_raw_path)
    print("Raw OOD data at:", ood_raw_path)

    id_path  = get_first_subdir(id_raw_path)
    ood_path = get_first_subdir(ood_raw_path)
    print("Adjusted ID path:", id_path)
    print("Adjusted OOD path:", ood_path)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    id_dataset  = datasets.ImageFolder(root=id_path,  transform=transform)
    ood_dataset = datasets.ImageFolder(root=ood_path, transform=transform)

    id_loader  = DataLoader(id_dataset,  batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)

    print(f"ID Dataset:  {len(id_dataset)} samples, {len(id_dataset.classes)} classes → {id_dataset.classes}")
    print(f"OOD Dataset: {len(ood_dataset)} samples, {len(ood_dataset.classes)} classes → {ood_dataset.classes}")

    return id_dataset, ood_dataset, id_loader, ood_loader

def save_sample_grid(loader, outpath, title="Samples", inv_norm=True):
    from torchvision.utils import make_grid
    images, labels = next(iter(loader))
    images = images[:8]
    if inv_norm:
        inv = transforms.Normalize(
            mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
            std=[1/s for s in [0.229, 0.224, 0.225]]
        )
        images = torch.stack([inv(img) for img in images])
    grid = make_grid(images, nrow=4)
    npimg = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(8, 4.5))
    plt.imshow(np.clip(npimg, 0, 1))
    plt.title(title); plt.axis('off')
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight"); plt.close()
