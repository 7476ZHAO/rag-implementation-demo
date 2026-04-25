from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import json

app = Flask(__name__)

# 1. Load Generation Model (Parametric Memory)
# FLAN-T5 is used as the generator for answering questions
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# 2. Load External Knowledge Base (Non-parametric Memory)
def load_docs():
    """Load external knowledge from a JSON file"""
    with open("data.json", "r") as f:
        return json.load(f)

docs = load_docs()

# 3. Initialize Embedding Model for Retrieval
# This approximates dense retrieval in the RAG paper
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute document embeddings for efficiency
doc_texts = [doc["text"] for doc in docs]
doc_embeddings = embedder.encode(doc_texts)

# 4. Retrieval Function (Semantic Search)
def retrieve(query):
    """
    Retrieve the most relevant document using semantic similarity.
    This replaces simple keyword matching with embedding-based retrieval.
    """
    query_embedding = embedder.encode(query)

    # Compute similarity scores (dot product)
    scores = np.dot(doc_embeddings, query_embedding)

    # Select the most relevant document
    best_idx = np.argmax(scores)

    return doc_texts[best_idx]

# 5. Generation WITHOUT RAG
def no_rag(query):
    """
    Generate an answer using only the model (parametric memory).
    No external knowledge is used.
    """
    prompt = f"""
Answer the question.

Question: {query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=False,
        repetition_penalty=2.0,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove repeated query if the model echoes it
    if query.lower() in answer.lower():
        answer = answer.replace(query, "").strip()

    return answer

# 6. Generation WITH RAG
def with_rag(query):
    """
    Generate an answer using both the query and retrieved context.
    This simulates Retrieval-Augmented Generation (RAG).
    """
    context = retrieve(query)

    prompt = f"""
Answer the question using the provided context.
Give a short and clear definition.

Context: {context}

Question: {query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=False,
        repetition_penalty=2.0,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove repeated query if the model echoes it
    if query.lower() in answer.lower():
        answer = answer.replace(query, "").strip()

    return answer

# 7. Flask Web Interface
@app.route("/", methods=["GET", "POST"])
def home():
    """Handle user input and return model response"""
    answer = ""

    if request.method == "POST":
        q = request.form.get("question")
        mode = request.form.get("mode")

        if not q:
            answer = "Please enter a question."
        else:
            if mode == "rag":
                answer = "With RAG: " + with_rag(q)
            else:
                answer = "Without RAG: " + no_rag(q)

    return render_template("index.html", answer=answer)

# 8. Run Application
if __name__ == "__main__":
    app.run(debug=True)