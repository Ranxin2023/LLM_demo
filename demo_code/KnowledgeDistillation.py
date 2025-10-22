import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------
# 1. Teacher and Student
# -----------------------
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# -----------------------
# 2. KD Loss Function
# -----------------------
def kd_loss(student_logits, teacher_logits, true_labels, T=2.0, alpha=0.7):
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    loss_kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    loss_ce = F.cross_entropy(student_logits, true_labels)
    return alpha * loss_kl + (1 - alpha) * loss_ce

# -----------------------
# 3. Training Step
# -----------------------
def train_knowledge_distillation():
    teacher = TeacherNet()
    student = StudentNet()
    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))

    with torch.no_grad():
        teacher_logits = teacher(x)

    student_logits = student(x)
    loss = kd_loss(student_logits, teacher_logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"KD Training step completed | Loss: {loss.item():.4f}")
