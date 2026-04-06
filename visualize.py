import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils.data import DataLoader
import config
from dataset import SegmentationDataset
from metrics import iou_score, pixel_accuracy

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
        raise ValueError(f"Invalid MODEL_NAME '{config.MODEL_NAME}'")


def iou_color(iou):
    """Return color based on IoU value — red=bad, yellow=ok, green=good."""
    if iou >= 0.7:
        return "#00e676"   # green
    elif iou >= 0.4:
        return "#ffea00"   # yellow
    else:
        return "#ff1744"   # red


def visualize(num_samples=6, split="val", save_path="runs/predictions.png"):
    # --- Load model ---
    model = get_model().to(config.DEVICE)
    checkpoint = torch.load(config.SAVE_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # --- Load dataset ---
    img_dir  = config.VAL_IMG_DIR  if split == "val"  else config.TRAIN_IMG_DIR
    mask_dir = config.VAL_MASK_DIR if split == "val"  else config.TRAIN_MASK_DIR

    dataset = SegmentationDataset(img_dir, mask_dir, image_size=config.IMAGE_SIZE)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    collected   = []
    all_iou     = []
    all_acc     = []

    with torch.no_grad():
        for image, mask in loader:
            image = image.to(config.DEVICE)
            mask  = mask.to(config.DEVICE)

            pred = model(image)
            if isinstance(pred, tuple):
                pred = pred[0]

            # --- Compute metrics BEFORE binarizing ---
            iou = iou_score(pred, mask)
            acc = pixel_accuracy(pred, mask)
            all_iou.append(iou)
            all_acc.append(acc)

            pred_sig    = torch.sigmoid(pred)
            pred_binary = (pred_sig > 0.5).float()

            # numpy for plotting
            img_np    = image[0].cpu().permute(1, 2, 0).numpy()
            mask_np   = mask[0, 0].cpu().numpy()
            pred_np   = pred_binary[0, 0].cpu().numpy()
            pred_conf = pred_sig[0, 0].cpu().numpy()

            collected.append((img_np, mask_np, pred_np, pred_conf, iou, acc))

            if len(collected) >= num_samples:
                break

    mean_iou = np.mean(all_iou)
    mean_acc = np.mean(all_acc)

    print(f"\n{'='*45}")
    print(f"  Results on {split} split ({len(collected)} samples)")
    print(f"{'='*45}")
    for i, (_, _, _, _, iou, acc) in enumerate(collected):
        status = "✅" if iou >= 0.5 else "⚠️ "
        print(f"  Sample {i+1:2d}  |  IoU: {iou:.4f}  |  Acc: {acc:.4f}  {status}")
    print(f"{'='*45}")
    print(f"  Mean IoU : {mean_iou:.4f}")
    print(f"  Mean Acc : {mean_acc:.4f}")
    print(f"{'='*45}\n")

    # --- Plot ---
    n   = len(collected)
    fig = plt.figure(figsize=(20, n * 4.2))
    fig.patch.set_facecolor("#0d0d1a")

    # 4 image cols + 1 narrow metrics col
    gs = gridspec.GridSpec(
        n, 5,
        figure=fig,
        hspace=0.4,
        wspace=0.08,
        width_ratios=[3, 3, 3, 3, 2]
    )

    col_titles = ["Input Image", "Ground Truth", "Prediction", "Confidence Map", "Metrics"]
    col_cmaps  = [None, "gray", "gray", "RdYlGn", None]

    for row, (img_np, mask_np, pred_np, pred_conf, iou, acc) in enumerate(collected):
        arrays = [img_np, mask_np, pred_np, pred_conf, None]

        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#0d0d1a")

            if col == 4:
                # --- Metrics panel ---
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

                color = iou_color(iou)

                # IoU badge
                ax.add_patch(plt.Rectangle((0.05, 0.55), 0.9, 0.35,
                             color="#1a1a2e", zorder=1, linewidth=0))
                ax.text(0.5, 0.80, "IoU Score", ha="center", va="center",
                        color="#aaaacc", fontsize=9, transform=ax.transAxes)
                ax.text(0.5, 0.63, f"{iou:.4f}", ha="center", va="center",
                        color=color, fontsize=18, fontweight="bold",
                        transform=ax.transAxes)

                # Acc badge
                ax.add_patch(plt.Rectangle((0.05, 0.10), 0.9, 0.35,
                             color="#1a1a2e", zorder=1, linewidth=0))
                ax.text(0.5, 0.36, "Pixel Acc", ha="center", va="center",
                        color="#aaaacc", fontsize=9, transform=ax.transAxes)
                ax.text(0.5, 0.19, f"{acc:.4f}", ha="center", va="center",
                        color="#4fc3f7", fontsize=18, fontweight="bold",
                        transform=ax.transAxes)

            else:
                if col == 0:
                    ax.imshow(np.clip(img_np, 0, 1))
                else:
                    ax.imshow(arrays[col], cmap=col_cmaps[col], vmin=0, vmax=1)

            if row == 0:
                ax.set_title(col_titles[col], color="white",
                             fontsize=12, fontweight="bold", pad=10)

            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", color="#888899",
                              fontsize=9, labelpad=6)

            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a4a")

    # Summary bar at top
    summary = (f"Model: {config.MODEL_NAME.upper()}  |  Split: {split}  |  "
               f"Mean IoU: {mean_iou:.4f}  |  Mean Acc: {mean_acc:.4f}")
    fig.suptitle(summary, color="white", fontsize=13,
                 fontweight="bold", y=1.01, x=0.5)

    plt.savefig(save_path, bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    visualize(
        num_samples=6,
        split="val",
        save_path="runs/predictions.png"
    )