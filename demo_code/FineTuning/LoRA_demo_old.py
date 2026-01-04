# pip install -U transformers peft datasets accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
def lora_demo():
    model_id = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    # Freeze backbone & add LoRA to attention projections
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["query","value"],  # names may be q_proj/v_proj on some models
        bias="none", task_type="SEQ_CLS"
    )
    model = get_peft_model(base, lora_cfg)

    ds = load_dataset("imdb", split={"train":"train[:2%]","test":"test[:2%]"})
    def tok_fn(b): return tok(b["text"], truncation=True)
    ds = {k:v.map(tok_fn, batched=True) for k,v in ds.items()}

    args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=16,
    per_device_eval_batch_size=16, num_train_epochs=2,
    evaluation_strategy="epoch", logging_steps=50, report_to="none"
    )
    trainer = Trainer(model=model, args=args,
                    train_dataset=ds["train"], eval_dataset=ds["test"],
                    tokenizer=tok)
    trainer.train()

    # Save tiny task head + LoRA only
    model.save_pretrained("adapters/sentiment")
