# Mini RAG Demo

## Overview
This project is a simplified implementation of Retrieval-Augmented Generation (RAG) for question answering.

It combines:
- **FLAN-T5** as the generator (parametric memory)
- **SentenceTransformer** for semantic retrieval
- A small **JSON knowledge base** (non-parametric memory)

The system demonstrates how retrieval improves answer quality compared to a model without external knowledge.

---

## Architecture
The pipeline follows the core idea of RAG:

query → embedding → retrieve → generate

- The query and knowledge texts are encoded into dense vector embeddings
- The most relevant text is retrieved using similarity search (dot product)
- The retrieved context is combined with the query for answer generation

---

## Features
- Compare **with RAG vs without RAG**
- Demonstrates reduction of hallucination
- Lightweight and easy to run locally

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the application
```bash
python app.py
```

### 3. Open in browser
```text
http://127.0.0.1:5000
```

## Example Questions
-What is RAG?
-Why do large language models need external knowledge?
-How does RAG improve accuracy?

## Notes
* This is a simplified prototype, not a full RAG system
* Uses a small dataset instead of large-scale sources like Wikipedia
* Retrieves only the top-1 document (no Top-K retrieval)
* No end-to-end training

## Relation to RAG Paper
This implementation is inspired by:
```
Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020.
```
It demonstrates the core idea of combining parametric memory (language model) with non-parametric memory (external knowledge).
