# Paper Similarity Ranking with Semantic Search

This Python project compares a given research paper with a collection of other papers, both locally stored and fetched from arXiv, based on their semantic similarity. The goal is to rank papers based on their relevance to the provided paper, leveraging a transformer-based model for text embeddings.

## Features

- Generate semantic embeddings for research papers using the `all-MiniLM-L6-v2` model.
- Calculate cosine similarity between paper embeddings to rank their relevance.
- Load papers from local directories (`.txt` files) and arXiv (PDFs).
- Automatically download and parse papers from arXiv based on keyword search queries.
- Visualize the similarity scores using a plot.

