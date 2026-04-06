import torch
from torch.utils.data import DataLoader
import config
from dataset import SegmentationDataset
from utils import load_checkpoint
from metrics import iou_score, pixel_accuracy
 
# FIX: single clean import block
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
        # FIX: was silently returning None before, causing cryptic crash later
        raise ValueError(f"Invalid MODEL_NAME '{config.MODEL_NAME}'.")
 
 
def main():
    model = get_model().to(config.DEVICE)
 
    # FIX: use load_checkpoint which handles map_location correctly
    model = load_checkpoint(config.SAVE_PATH, model, config.DEVICE)
    model.eval()
 
    test_ds = SegmentationDataset(
        config.TEST_IMG_DIR,
        image_size=config.IMAGE_SIZE,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
 
    all_iou = []
    all_acc = []
 
    with torch.no_grad():
        for idx, x in enumerate(loader):
            x = x.to(config.DEVICE)
            preds = model(x)
 
            if isinstance(preds, tuple):
                preds = preds[0]
 
            preds = torch.sigmoid(preds)
            binary_preds = (preds > 0.5).float()
 
            print(f"[{idx+1}/{len(loader)}] Output shape: {preds.shape}  "
                  f"Predicted positives: {binary_preds.sum().item():.0f}")
 
    print("\nInference complete.")
 
 
if __name__ == "__main__":
    main()