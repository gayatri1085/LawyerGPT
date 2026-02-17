# LawyerGPT – AI Legal Assistant using RAG

## Overview
LawyerGPT is an AI-powered legal assistant built using **Retrieval-Augmented Generation (RAG)**.  
It answers legal questions by retrieving relevant sections from legal documents (e.g., Indian Constitution, IPC, contracts) and generating accurate, context-aware responses.

Unlike generic chatbots, LawyerGPT provides **grounded legal answers** by citing real legal sources.

---

## Features
- Upload and index legal documents (PDF/TXT)
- Semantic search using embeddings
- Context-aware legal answers
- Supports Indian legal domain (can be extended globally)
- Custom knowledge base
- Source-aware responses

---

## Use Cases
- Law students for case analysis  
- Legal researchers  
- Contract review assistance  
- Constitutional reference system  
- Legal chatbots for websites  

---

## Tech Stack
- Python  
- Transformers (LLMs)  
- Sentence Transformers  
- FAISS (Vector search)  
- LangChain (optional)  

---

## Installation

```bash
git clone https://github.com/yourusername/LawyerGPT.git
cd LawyerGPT
pip install -r requirements.txt
