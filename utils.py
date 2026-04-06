import torch
from tqdm import tqdm
 
 
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader)
    total_loss = 0
 
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)
 
        predictions = model(data)
 
        # Handle models that return tuples (e.g. DeepLabV3+)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
 
        loss = loss_fn(predictions, targets)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
 
    return total_loss / len(loader)
 
 
def val_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
 
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
 
            predictions = model(data)
 
            # Handle models that return tuples
            if isinstance(predictions, tuple):
                predictions = predictions[0]
 
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
 
    model.train()
    return total_loss / len(loader)
 
 
def save_checkpoint(model, filename):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(model.state_dict(), filename)
 
 
def load_checkpoint(filename, model, device):
    """FIX: added map_location so it works on CPU machines too."""
    print(f"=> Loading checkpoint from {filename}")
    model.load_state_dict(torch.load(filename, map_location=device))
    return model