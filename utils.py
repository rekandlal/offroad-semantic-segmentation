import torch
from tqdm import tqdm


def train_fn(loader, model, optimizer, loss_fn, device):
    loop = tqdm(loader)

    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        predictions = model(data)
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
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

    model.train()
    return total_loss / len(loader)


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)