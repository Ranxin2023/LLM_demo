from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
# -----------------------
# 1. Teacher and Student
# -----------------------
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

    def forward(self, x):
        # Layer 1: ReLU + normalization + dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        # Layer 2: ReLU + normalization + dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        # Layer 3: ReLU + normalization + dropout
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        # Final output logits (no softmax here)
        return self.fc4(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------
# 2. KD Loss Function
# -----------------------
def kd_loss(student_logits, teacher_logits, true_labels, T=3.5, alpha=0.6):
    """Smoothed knowledge distillation loss."""
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    loss_kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    loss_ce = F.cross_entropy(student_logits, true_labels)
    return alpha * loss_kl + (1 - alpha) * loss_ce


# -----------------------
# 3. Training Step
# -----------------------

teacher = TeacherNet()
student = StudentNet()


def train_knowledge_distillation_lowest(teacher, student):
    optimizer = optim.AdamW(student.parameters(), lr=6e-4, weight_decay=1e-5)
    # One-Cycle schedule: start small, peak mid-training, decay to near 0
    scheduler = OneCycleLR(
        optimizer,
        max_lr=6e-4,
        epochs=150,
        steps_per_epoch=16,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=20,
        final_div_factor=100
    )

    for p in teacher.parameters():
        p.requires_grad = False

    x = torch.randn(1024, 784)
    y = torch.randint(0, 10, (1024,))

    num_epochs = 150
    batch_size = 64

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(x) // batch_size

        for i in range(0, len(x), batch_size):
            inputs = x[i:i+batch_size]
            labels = y[i:i+batch_size]

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            loss = kd_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] | KD Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

    torch.save(student.state_dict(), "distilled_student_model_lowest.pth")
    print("\n✅ Training completed! Model saved as distilled_student_model_lowest.pth")

def knowledge_distillation_redirect():
    with open("./output_results/knowledge_distillation.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            train_knowledge_distillation_lowest(teacher=teacher, student=student)