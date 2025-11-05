import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import redirect_stdout

# ------------------------------
# 1️⃣ Define Expert Networks
# ------------------------------
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.layer(x)  # output scalar per sample


# ------------------------------
# 2️⃣ Define the Gating Network
# ------------------------------
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Compute probability for each expert
        gate_logits = self.fc(x)
        gate_weights = F.softmax(gate_logits, dim=1)
        return gate_weights


# ------------------------------
# 3️⃣ Combine into Mixture of Experts
# ------------------------------
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # Get expert weights from gate
        weights = self.gate(x)  # shape (batch, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # (batch, 1, num_experts)
        output = torch.bmm(expert_outputs, weights.unsqueeze(2))  # weighted sum across experts
        return output.squeeze(2)  # (batch, 1) → (batch,)


# ------------------------------
# 4️⃣ Example Usage
# ------------------------------
def moe_demo():
    torch.manual_seed(0)

    model = MixtureOfExperts(input_dim=2, hidden_dim=8, num_experts=3)
    x = torch.randn(5, 2)  # 5 samples, 2 features each
    output = model(x)

    print("Input:\n", x)
    print("\nOutput:\n", output)

def moe_demo_redirect():
    with open("./output_results/moe_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            moe_demo()