# from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification
# from demo_code.Agent_demo.generate_knowledge_base import generate_knowledge_base
# def fine_tune():
#     model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#     training_args = TrainingArguments(
#         output_dir="./results", per_device_train_batch_size=8, num_train_epochs=3
#     )
#     trainer = Trainer(model=model, args=training_args, train_dataset=your_domain_dataset)
#     trainer.train()
# implement later