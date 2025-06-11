from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

def perform_RAG():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

    question = "What is the theory of relativity?"
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Perform retrieval + generation
    generated = model.generate(input_ids=input_ids)
    answer = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print("RAG Answer:", answer[0])