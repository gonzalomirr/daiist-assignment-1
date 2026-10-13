"""
The one model (L2-regularized logistic regression) written for the two
PyTorch implementations, plus the inference helpers app.py uses to score
with the *saved* weights. scikit-learn's version needs no code here.

All three implementations minimise the same objective, so on the same data
they should land on (almost) the same coefficients:

    mean_i  BCE(y_i, sigmoid(w.x_i + b))  +  (lam / 2) * ||w||^2

scikit-learn writes it as  0.5 * ||w||^2 + C * sum_i BCE(...); dividing by
C * n gives the form above with lam = 1 / (C * n). The bias b is not
penalized in any of the three.
"""

import numpy as np
import torch
from torch import nn


def sklearn_C_to_lambda(C: float, n_train: int) -> float:
    return 1.0 / (C * n_train)


# ---------------------------------------------------------------------------
# Manual version: raw tensors + autograd + a hand-written gradient step.
# ---------------------------------------------------------------------------
def manual_bce_with_l2(X: torch.Tensor, y: torch.Tensor,
                       w: torch.Tensor, b: torch.Tensor, lam: float) -> torch.Tensor:
    logits = X @ w + b
    # BCE written out by hand. -log(sigmoid(z)) = softplus(-z) and
    # -log(1 - sigmoid(z)) = softplus(z) are the numerically stable forms of
    # -[y log p + (1 - y) log(1 - p)]: no log(0) when p saturates.
    bce = (y * nn.functional.softplus(-logits)
           + (1 - y) * nn.functional.softplus(logits)).mean()
    return bce + 0.5 * lam * (w ** 2).sum()


def train_manual(X: np.ndarray, y: np.ndarray, lam: float,
                 lr: float, epochs: int, seed: int = 0):
    """Full-batch gradient descent, Session 5 style. Returns (w, b, losses)."""
    g = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = (0.01 * torch.randn(X.shape[1], generator=g)).requires_grad_()
    b = torch.zeros(1, requires_grad=True)

    losses = []
    for _ in range(epochs):
        loss = manual_bce_with_l2(Xt, yt, w, b, lam)
        loss.backward()                     # autograd fills w.grad and b.grad
        with torch.no_grad():               # the update itself is not part of the graph
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                      # .backward() accumulates, so reset
        b.grad.zero_()
        losses.append(loss.item())
    return w.detach(), b.detach(), losses


def predict_manual(params: dict, X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(Xt @ params["w"] + params["b"]).numpy()


# ---------------------------------------------------------------------------
# Standard version: nn.Module + torch.optim + DataLoader.
# ---------------------------------------------------------------------------
class LogisticRegressionNet(nn.Module):
    """A single linear layer. It outputs *logits*; the sigmoid lives in the
    loss (BCEWithLogitsLoss) during training and in predict() afterwards."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_standard(X: np.ndarray, y: np.ndarray, lam: float, lr: float,
                   epochs: int, batch_size: int | None = 512, seed: int = 0):
    """Mini-batch Adam. Returns (model, per-epoch full-train losses)."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    if batch_size is None:
        # Full batch (used only for a diagnostic in train.py). DataLoader
        # would re-collate all rows one by one on every step, so skip it.
        loader = [(Xt, yt)]
    else:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    model = LogisticRegressionNet(X.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    # weight_decay adds lam * w to the gradient, i.e. the gradient of
    # (lam / 2) * ||w||^2 -- the same L2 term as above. Put on the weights
    # only, so the bias stays unpenalized like in scikit-learn.
    optimizer = torch.optim.Adam([
        {"params": [model.linear.weight], "weight_decay": lam},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)

    losses = []
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():   # log the same full objective as the manual loop
            w = model.linear.weight.squeeze(0)
            full = loss_fn(model(Xt), yt) + 0.5 * lam * (w ** 2).sum()
        losses.append(full.item())
    return model, losses


def predict_standard(model: LogisticRegressionNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
