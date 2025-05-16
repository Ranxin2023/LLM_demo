
from contextlib import redirect_stdout

from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, accuracy_score
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from demo_code.init_openai import query_gpt4
import math
import sacrebleu
import torch

def evaluate_gpt_two():
    print("Evaulate gpt 2......")
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    # Example input
    input_text = "The capital of France is"
    true_continuation = " Paris"
    true_classification = 1  # True label for demonstration
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        logits = outputs.logits
    # Compute loss by forwarding the same input as label
    with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
    def calculate_perplexity(loss):
        return math.exp(loss)

    print("Perplexity:", calculate_perplexity(loss.item()))
    # 2. Text Generation
    generator = pipeline("text-generation", model="gpt2")
    generated = generator(input_text, max_length=10)[0]["generated_text"]
    print("Generated Text:", generated)
    # 3. Accuracy (simulated task)
    y_pred = [1]
    y_true = [true_classification]
    print("Accuracy:", accuracy_score(y_true, y_pred))
    # 4. F1 Score (binary classification)
    print("F1 Score:", f1_score(y_true, y_pred, average="binary"))
    def compute_bleu(reference, hypothesis):
        """
                Compute BLEU score using sacrebleu (compatible with Python 3.12).
                Both reference and hypothesis should be lists of tokens.
        """
        reference = [" ".join(reference)]
        hypothesis = " ".join(hypothesis)
        result = sacrebleu.sentence_bleu(hypothesis, reference)
        return result.score
    # 5. BLEU Score
    reference = true_continuation.strip().split()
    hypothesis = generated.replace(input_text, "").strip().split()

    print("BLEU Score:", compute_bleu(reference, hypothesis))
    # 6. ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(true_continuation.strip(), generated.replace(input_text, "").strip())
    print("ROUGE Scores:", rouge_scores)

def evaluate_gpt_four():
    print("Evaluating gpt 4.......")
    def estimate_perplexity(output_text):
        # Simulate perplexity by estimating based on token count (for demonstration only)
        token_count = len(output_text.split())
        # More tokens = more entropy; smaller result = lower perplexity
        return math.exp(token_count / 2)  # Not mathematically accurate, but illustrative
    
    def calculate_accuracy_and_f1(expected, actual):
        # Simulate binary classification (correct = 1, incorrect = 0)
        y_true = [1]
        y_pred = [1 if expected.lower() in actual.lower() else 0]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, f1
    
    def evaluate_output(reference, generated):
        reference = [" ".join(reference.strip().split())]
        hypothesis = " ".join(generated.strip().split())

        bleu = sacrebleu.sentence_bleu(hypothesis, reference).score
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True).score(
            reference[0], hypothesis
        )
        return bleu, rouge
    prompt = "What is the capital of France?"
    expected = "Paris is the capital of France."

    gpt4_output = query_gpt4(prompt, temperature=0.7)

    # different scores
    bleu_score, rouge_scores = evaluate_output(expected, gpt4_output)
    accuracy, f1 = calculate_accuracy_and_f1(expected, gpt4_output)
    perplexity = estimate_perplexity(gpt4_output)

    # print part
    print("🔹 GPT-4 Output:", gpt4_output)
    print("🔸 Perplexity (simulated):", perplexity)
    print("🔸 Accuracy:", accuracy)
    print("🔸 F1 Score:", f1)
    print("🔸 BLEU Score:", bleu_score)
    print("🔸 ROUGE Scores:", rouge_scores)

def evaluate_LLM():
    with open("./output_results/evaluate_LLM.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            evaluate_gpt_two()
            evaluate_gpt_four()



