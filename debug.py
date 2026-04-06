"""
Run this FIRST before training to verify your dataset is correct.
Usage: python debug.py
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import config


def check_dataset():
    mask_dir = config.TRAIN_MASK_DIR
    img_dir  = config.TRAIN_IMG_DIR

    mask_files = sorted(os.listdir(mask_dir))
    img_files  = sorted(os.listdir(img_dir))

    print(f"Total train images : {len(img_files)}")
    print(f"Total train masks  : {len(mask_files)}")
    print()

    # Check 5 random masks
    sample_files = random.sample(mask_files, min(5, len(mask_files)))

    all_white = 0
    all_black = 0
    mixed     = 0

    for fname in mask_files[:200]:  # check first 200
        mask_path = os.path.join(mask_dir, fname)
        mask = np.array(Image.open(mask_path).convert("L"))
        unique = np.unique(mask)

        if len(unique) == 1 and unique[0] == 255:
            all_white += 1
        elif len(unique) == 1 and unique[0] == 0:
            all_black += 1
        else:
            mixed += 1

    print(f"Out of first 200 masks:")
    print(f"  All white (255) : {all_white}  ← BAD if this is all of them")
    print(f"  All black (0)   : {all_black}  ← BAD if this is all of them")
    print(f"  Mixed (correct) : {mixed}   ← GOOD, these have actual segmentation")
    print()

    if all_white == 200:
        print("⚠️  WARNING: ALL your masks are solid white!")
        print("   This means your mask files may be wrong/corrupted.")
        print("   The model is learning to predict everything as foreground.")
    elif mixed == 0:
        print("⚠️  WARNING: No masks have mixed content — check your mask folder.")
    else:
        print("✅ Masks look correct — mixed content found.")

    # --- Visual check: show 6 image+mask pairs ---
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Dataset Debug — Top row: Images | Bottom row: Masks",
                 color="white", fontsize=13)

    for i, fname in enumerate(random.sample(mask_files, 6)):
        img_path  = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_np = np.array(mask)
        unique_vals = np.unique(mask_np)

        ax_img  = axes[0][i]
        ax_mask = axes[1][i]

        ax_img.imshow(img)
        ax_img.set_title(fname[:12], color="white", fontsize=7)
        ax_img.axis("off")

        ax_mask.imshow(mask_np, cmap="gray", vmin=0, vmax=255)
        ax_mask.set_title(f"vals: {unique_vals}", color="yellow", fontsize=7)
        ax_mask.axis("off")

    plt.tight_layout()
    plt.savefig("runs/debug_dataset.png", dpi=120, facecolor="#1a1a2e")
    plt.show()
    print("\nSaved debug image to: runs/debug_dataset.png")


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    check_dataset()