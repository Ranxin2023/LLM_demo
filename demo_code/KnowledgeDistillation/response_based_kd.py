from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----- Teacher and Student -----
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

teacher = TeacherNet()
student = StudentNet()

# ----- Response-Based KD Loss -----
def kd_loss(student_logits, teacher_logits, true_labels, T=3.0, alpha=0.6):
    """Classic Response-Based Knowledge Distillation"""
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    loss_kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    loss_ce = F.cross_entropy(student_logits, true_labels)
    return alpha * loss_kl + (1 - alpha) * loss_ce

def train_response_based_kd(teacher, student):
    """Train student using response-based knowledge distillation."""
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    
    # Dummy training data (replace with real dataset in practice)
    x = torch.randn(1024, 784)
    y = torch.randint(0, 10, (1024,))
    
    num_epochs = 50
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
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] | KD Loss: {avg_loss:.4f}")

    torch.save(student.state_dict(), "distilled_student_response_based.pth")
    print("\n✅ Response-based KD training completed! Model saved as distilled_student_response_based.pth")


def response_based_redirect():
    """Redirect response-based KD training logs to output file."""
    with open("./output_results/response_based_kd.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            train_response_based_kd(teacher=teacher, student=student)