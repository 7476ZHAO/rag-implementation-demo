# RAG Demo

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

- The query and knowledge texts are encoded into vector embeddings
- The most relevant text is retrieved using similarity search (dot product)
- The retrieved context is combined with the query for answer generation

---

## Features
- Compare **with RAG vs without RAG**
- Demonstrates reduction of hallucination
- Lightweight and easy to run locally

---

## Installation

```bash
pip install -r requirements.txt
