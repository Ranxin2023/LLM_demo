# -----------------------------
# 1. Imports
# -----------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup
)

from torch.optim import AdamW

# -----------------------------
# 2. Custom Medical Dataset
# -----------------------------

class MedicalQADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"Q: {item['instruction']}\nA:"
        answer = item["output"]

        text = prompt + " " + answer

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc

    def __len__(self):
        return len(self.data)
    
# -----------------------------
# 3. Load Tokenizer & Model
# -----------------------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Full fine-tuning:
for param in model.parameters():
    param.requires_grad = True

# -----------------------------
# 4. Prepare Dataset & Loader
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')

import json
dataset_path = "/content/drive/MyDrive/Colab Notebooks/LLM_demo/datasets/medical_dataset.jsonl"
with open(dataset_path, "r") as f:
    dataset_raw = [json.loads(l) for l in f]

dataset = MedicalQADataset(dataset_raw, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# -----------------------------
# 5. Optimizer & Scheduler
# -----------------------------
epochs = 100
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

num_training_steps = len(loader) * epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

model.cuda()

# -----------------------------
# 6. Training Loop
# -----------------------------
import os
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]
            )
            loss = outputs.loss

        scaler.scale(loss).backward()

        # Gradient clipping (important for LLMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    # -------------------------
    # 6. Save checkpoint
    # -------------------------
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

# -----------------------------
# 7. Testing Models
# -----------------------------

### 1. Load Your Fine-Tuned Model + Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Path to your saved checkpoint (modify as needed)
checkpoint_path = "/content/checkpoints/epoch_99.pth"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load pre-trained GPT2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load your fine-tuned weights
model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
model = model.cuda()
model.eval()

print("Model loaded successfully!")

### 2. Write a Helper Function to Ask Questions
def ask_medical_question(question, max_new_tokens=100):
    prompt = f"Q: {question}\nA:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

print(ask_medical_question("What are symptoms of anemia?"))
print(ask_medical_question("How do doctors treat viral infections?"))
print(ask_medical_question("What causes chest pain?"))
print(ask_medical_question("How is dehydration treated?"))
print(ask_medical_question("What are signs of kidney failure?"))
print(ask_medical_question("Explain how fever works."))
'''
output results:
    Q: What are symptoms of anemia?
    A: Fatigue, pale skin, dizziness, shortness of breath, cold hands and feet.
    Q: How do doctors treat viral infections?
    A: Viruses, bacteria, or fungi infecting the lungs.
    Q: What causes chest pain?
    A: Injury, infection, autoimmune diseases like rheumatoid arthritis, or overuse.
    Q: How is dehydration treated?
    A: Rest, fluids, fever reducers, and oxygen therapy when needed.
    Q: What are signs of kidney failure?
    A: Fatigue, pale skin, dizziness, shortness of breath, cold hands and feet.
    Q: Explain how fever works.
    A: It causes headache, fatigue, dizziness, dry mouth, and can lead to low blood pressure or heat stroke.
'''