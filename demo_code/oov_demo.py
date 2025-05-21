from contextlib import redirect_stdout
from transformers import BertTokenizer, GPT2Tokenizer
def oov_demo():
    with open("./output_results/output_control.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Example OOV-like word
            word = "transformerssaurus"  # Not in vocabulary

            # BERT uses WordPiece
            print("\n🔹 Tokenization using BERT (WordPiece):")
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokens_bert = bert_tokenizer.tokenize(word)
            print(f"Input: {word}")
            print(f"Tokens: {tokens_bert}")

            # GPT-2 uses Byte Pair Encoding (BPE)
            print("\n🔹 Tokenization using GPT-2 (BPE):")
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokens_gpt2 = gpt2_tokenizer.tokenize(word)
            print(f"Input: {word}")
            print(f"Tokens: {tokens_gpt2}")
