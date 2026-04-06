import torch

def iou_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection

    return (intersection / (union + 1e-6)).item()


def pixel_accuracy(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    correct = (preds == targets).float().sum()
    total = torch.numel(preds)

    return (correct / total).item()