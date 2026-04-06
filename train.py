import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from dataset import SegmentationDataset

# models
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormer
from models.unetplusplus import UNetPlusPlus
from deeplabv3plus import DeepLabV3Plus
from segformer import SegFormer
from unetplusplus import UNetPlusPlus
from loss import BCEDiceLoss


def get_model():
    if config.MODEL_NAME == "deeplab":
        return DeepLabV3Plus()
    elif config.MODEL_NAME == "segformer":
        return SegFormer()
    elif config.MODEL_NAME == "unetpp":
        return UNetPlusPlus()
    else:
        raise ValueError("Invalid model")


def train_fn(loader, model, optimizer, loss_fn):
    model.train()

    for x, y in loader:
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        preds = model(x)

        # handle tuple outputs
        if isinstance(preds, tuple):
            preds = preds[0]

        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    model = get_model().to(config.DEVICE)

    # ✅ FIXED LOSS
    loss_fn = BCEDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_ds = SegmentationDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)
        print(f"Epoch {epoch+1} completed")

    torch.save(model.state_dict(), config.SAVE_PATH)


if __name__ == "__main__":
    main()