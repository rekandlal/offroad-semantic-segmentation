class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Args:
        patience:  How many epochs to wait after last improvement.
        min_delta: Minimum decrease in val_loss to count as improvement.
    """
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found — reset counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            print(f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                print("  => Early stopping triggered.")