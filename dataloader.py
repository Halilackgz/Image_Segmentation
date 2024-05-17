from torch.utils.data import Dataset, DataLoader
import torch

import os
import cv2
from PIL import Image
import numpy as np


class loader(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.files = os.listdir(os.path.join(self.path, "original"))
        self.masks = os.listdir(os.path.join(self.path, "mask"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        print(self.files[index])
        image = np.array(Image.open(os.path.join(self.path, "original", self.files[index])))
        mask = np.array(Image.open(os.path.join(self.path, "mask", self.files[index])))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imshow("mask", cv2.resize(mask, (256, 256)))
        cv2.imshow("image", cv2.resize(image, (256, 256)))

        image = torch.tensor(cv2.resize(image, (256, 256))/255., dtype=torch.float32)
        mask = torch.tensor(cv2.resize(mask, (256, 256))>128, dtype=torch.float32)
        
        
        image = image.permute((2, 0, 1))
        mask = mask.unsqueeze(0)
        
        return image, mask
    
if __name__ == "__main__":
    train_dataset = loader("/home/mr4t/Desktop/halil_segmentation/BCCD Dataset with mask/train")

    for image, mask in DataLoader(train_dataset, batch_size=1, shuffle=True):
        
        print(mask, mask.shape)
        break
