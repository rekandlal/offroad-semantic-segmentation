import torch
from torch.utils.data import DataLoader
import config
from dataset import SegmentationDataset
from loss import BCEDiceLoss
from utils import train_fn, val_fn, save_checkpoint
from early_stopping import EarlyStopping

from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormer
from models.unetplusplus import UNetPlusPlus


def get_model():
    if config.MODEL_NAME == "deeplab":
        return DeepLabV3Plus()
    elif config.MODEL_NAME == "segformer":
        return SegFormer()
    elif config.MODEL_NAME == "unetpp":
        return UNetPlusPlus()
    else:
        raise ValueError(f"Invalid MODEL_NAME '{config.MODEL_NAME}'. Choose: deeplab, segformer, unetpp")


def main():
    print(f"Using device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}  |  Image size: {config.IMAGE_SIZE}  |  Batch: {config.BATCH_SIZE}")

    model = get_model().to(config.DEVICE)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler — reduces LR when stuck, helps on CPU
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_ds = SegmentationDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        image_size=config.IMAGE_SIZE,
    )
    val_ds = SegmentationDataset(
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        image_size=config.IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"Steps per epoch: {len(train_loader)}")
    print("-" * 50)

    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
    )

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, config.DEVICE)
        val_loss = val_fn(val_loader, model, loss_fn, config.DEVICE)

        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Step the LR scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss <= early_stopping.best_loss:
            save_checkpoint(model, config.SAVE_PATH)

        # Check early stopping
        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"\nStopped early at epoch {epoch+1}. Best val loss: {early_stopping.best_loss:.4f}")
            break

    print(f"\nTraining complete. Best model saved to: {config.SAVE_PATH}")


if __name__ == "__main__":
    main()