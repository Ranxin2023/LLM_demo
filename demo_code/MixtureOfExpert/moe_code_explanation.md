# Explanation of `moe_demo.py`
## Step 1: Define Expert Networks
```python
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.layer(x)

```
- **What it does**:
    - Each Expert is an **independent neural network** that performs a small task (a sub-function).
    - It receives an input vector (e.g., a token embedding or image features).
    - Processes it through two linear layers with a **ReLU** activation in between.
- **Why it's important**:
    - Each expert learns to specialize in a **subset of patterns**.
    - Example: In a text model, one expert may learn syntax, another sentiment, another logic.
- **Output**
    - Each expert produces a single output value per input sample (for simplicity in this demo).
    - In real MoE models, experts return **vector embeddings**, not scalars.

## Step 2: Define the Gating Network

```python
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_logits = self.fc(x)
        gate_weights = F.softmax(gate_logits, dim=1)
        return gate_weights


```
- **What it does**:
    - The **gating network** decides **which experts to activate** for each input.
    - Input: same features as the experts (e.g., token representation).
    - Output: a probability distribution over all experts — i.e.,how much weight to assign to each expert’s opinion.
- **Mathematics**:
    $$
    w = \text{softmax}(W_g x)
    $$
    - where
        - 𝑊𝑔 = weights of the gating network
        - 𝑥 = input vector
        - 𝑤 = normalized weights for each expert (sum to 1)
## Step 3: Combine Experts into MoE Model
```python
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)

```
- We create multiple experts (e.g., 3) and a single gating network.
- These together form the **MoE layer**.
- 