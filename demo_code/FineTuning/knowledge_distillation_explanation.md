# `KnowledgeDistilation.py` Explanation
## File Overview
- This script builds a **Teacher–Student model training process** using Knowledge Distillation (KD).
It includes:
1. Definition of the **Teacher** and **Student** neural networks.
2. The **KD loss function**, which mixes Cross-Entropy and KL Divergence.
3. A **training loop** to transfer knowledge from teacher → student.
4. A **helper function** to redirect output to a file.

## 1. TeacherNet class
```python
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer (for 28x28 flattened images → 784 features)
        self.fc1 = nn.Linear(784, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.3)

        # Deep hidden layer 1
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.3)

        # Deep hidden layer 2
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.2)

        # Output layer
        self.fc4 = nn.Linear(512, 10)

```
- This is the teacher model — a large, powerful network that already knows how to perform a classification task.
- It serves as the knowledge source for distillation.
### Components

| Layer                  | Function        | Explanation                                                                 |
| ---------------------- | --------------- | --------------------------------------------------------------------------- |
| `nn.Linear(784, 2048)` | Fully connected | Takes a flattened 28×28 input (784 pixels) and projects it to 2048 features |
| `nn.BatchNorm1d(2048)` | Normalization   | Stabilizes learning and speeds up convergence                               |
| `nn.Dropout(0.3)`      | Regularization  | Prevents overfitting by randomly dropping neurons                           |
| `fc2`, `fc3`           | Deeper layers   | Capture hierarchical representations                                        |
| `fc4`                  | Output layer    | Produces logits for 10 classes (e.g., digits 0–9)                           |

### Forward Pass
```python
def forward(self, x):
    x = F.relu(self.bn1(self.fc1(x)))
    x = self.dropout1(x)
    x = F.relu(self.bn2(self.fc2(x)))
    x = self.dropout2(x)
    x = F.relu(self.bn3(self.fc3(x)))
    x = self.dropout3(x)
    return self.fc4(x)

```
## 3. `kd_loss()` Function
```python
def kd_loss(student_logits, teacher_logits, true_labels, T=3.5, alpha=0.6):
    """Smoothed knowledge distillation loss."""
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    loss_kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    loss_ce = F.cross_entropy(student_logits, true_labels)
    return alpha * loss_kl + (1 - alpha) * loss_ce

```
- This defines the Knowledge Distillation loss, which combines:
    - **KL Divergence (teacher–student imitation)**
    - **Cross-Entropy (true label learning)**
### Parameters
| Parameter        | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `T`              | Temperature (3.5 softens probabilities to reveal “dark knowledge”) |
| `alpha`          | Weight between teacher imitation and ground-truth learning         |
| `teacher_logits` | Raw outputs from teacher model                                     |
| `student_logits` | Student’s outputs                                                  |
| `true_labels`    | Actual labels (for supervised learning)                            |
## 4. train_knowledge_distillation_lowest() Function
