from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SurgicalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list=None, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.image_dir.glob("*.png")))
        if image_list:
            self.image_paths = [self.image_dir / fname for fname in image_list]

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((1024, 1024), interpolation=Image.BILINEAR)
        self.resize_mask = transforms.Resize((1024, 1024), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = Path(str(img_path).replace("images", "masks"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = self.resize(image)
            mask = self.resize_mask(mask)
            image = self.to_tensor(image)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask, img_path.name
