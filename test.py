import torch
from torch.utils.data import DataLoader
import config
from dataset import SegmentationDataset

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

def main():
    model = get_model().to(config.DEVICE)
    model.load_state_dict(torch.load(config.SAVE_PATH))
    model.eval()

    test_ds = SegmentationDataset(config.TEST_IMG_DIR)
    loader = DataLoader(test_ds, batch_size=1)

    with torch.no_grad():
        for x in loader:
            x = x.to(config.DEVICE)
            preds = model(x)

            if isinstance(preds, tuple):
                preds = preds[0]

            preds = torch.sigmoid(preds)
            print(preds.shape)

if __name__ == "__main__":
    main()