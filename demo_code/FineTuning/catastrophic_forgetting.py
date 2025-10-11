# Demo: Catastrophic Forgetting (Toy Example)
from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simple model: single hidden layer MLP
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Task A: learn y = x^2
# Task B: learn y = sin(x)
def train(model, fn, epochs=500):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        x = torch.linspace(-3, 3, 100).unsqueeze(1)
        y = fn(x)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, fn):
    x = torch.linspace(-3, 3, 100).unsqueeze(1)
    y = fn(x)
    pred = model(x).detach()
    return x, y, pred

def catastrophic_forgetting():
    # Train on Task A, then Task B
    model = SimpleNN()

    # Step 1: Train on y = x^2
    train(model, lambda x: x**2)
    x, y_true, y_pred_A = evaluate(model, lambda x: x**2)

    # Step 2: Fine-tune on y = sin(x)
    train(model, torch.sin)
    x, y_true_sin, y_pred_B = evaluate(model, lambda x: x**2)

    # Plot forgetting effect
    plt.figure(figsize=(10,5))
    plt.plot(x, y_true, label="True y = x^2 (Task A)")
    plt.plot(x, y_pred_A, '--', label="After training Task A")
    plt.plot(x, y_pred_B, ':', label="After fine-tuning Task B (forgot Task A)")
    plt.legend()
    plt.title("Catastrophic Forgetting Example")
    plt.show()

def catastrophic_forgetting_redirect():
    with open("./output_results/catastrophic_forgetting.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            catastrophic_forgetting()