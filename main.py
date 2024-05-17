from unet.unet_model import UNet
import torch
from torch.optim import Adam
from torch.nn import MSELoss

from dataloader import loader
from torch.utils.data import DataLoader

from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCH = 100

train_dataset = loader("/home/mr4t/Desktop/halil_segmentation/BCCD Dataset with mask/train")
val_dataset = loader("/home/mr4t/Desktop/halil_segmentation/BCCD Dataset with mask/test")

model = UNet(3, 1, True)
model = model.to(device)

criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCH):
    train_loss = 0
    val_loss = 0
    for image, mask in tqdm(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)):
        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        pred = model(image)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

    for image, mask in DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True):
        image = image.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            pred = model(image)
            loss = criterion(pred, mask)

            val_loss += loss.item()

    print("Train Loss: ", train_loss/(len(train_dataset)//BATCH_SIZE), "Val Loss: ", val_loss/(len(val_dataset)//BATCH_SIZE))
    if epoch%5 == 0:
        torch.save(model.state_dict(), f"model_{epoch}.pt")