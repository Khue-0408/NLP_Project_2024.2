# Legal Retrieval: Contrastive E5 Embedding Pipeline

This repository contains everything you need to build and evaluate a dense+contrastive embedding retrieval pipeline on legal text.  

You'll find:
- A **Contrastive_E5** folder with scripts and notebooks to embed your corpus, mine hard negatives, create training data, and train your own embedding model.  
- Standalone notebooks for classic IR baselines: **BM25**, **BM25F**, **FAISS**.  
- A **Preprocess.ipynb** for all data cleaning and tokenization steps.  

> **Note:** We plan to add a cross-encoder reranker soon. For now you can ignore reranker references.

---

## 📂 Folder Structure

```
.
├── Contrastive_E5/
│   ├── create_training_data.py
│   ├── encoder_legal_corpus_embedding.ipynb
│   ├── encoder_legal_train_embedding.ipynb
│   ├── mine_hard.py
│   ├── requirements.txt
│   └── train_embedding.py
├── BM25.ipynb
├── BM25F.ipynb
├── FAISS.ipynb
└── Preprocess.ipynb
```

---

## 🚀 Quick Start

### 1. Prepare your data
- **`corpus.csv`** – each row is one legal document (ID + text).  
- **`train.csv`** – each row is one QA pair (ID + question + answer passage ID).

Place them at the root of this repo.

### 2. Embed your documents & train set

1. **Embed the full corpus**  
   Open and run `Contrastive_E5/encoder_legal_corpus_embedding.ipynb`.  
   - Output:  
     - `full_emb.npy` (N × D float32 embeddings)  
     - `corpus_ids.npy` (N document IDs)  

2. **Embed the train set**  
   Open and run `Contrastive_E5/encoder_legal_train_embedding.ipynb`.  
   - Output:  
     - `train_emb.npy` (M × D float32 embeddings)  
     - `train_ids.npy` (M train-question IDs)  

> ⚠️ If you want to use our reference weights, download them here and update the paths in the notebooks:  
> https://huggingface.co/models/legal-embeddings

### 3. Mine hard negatives

```bash
cd Contrastive_E5
python mine_hard.py
```

* Input: `full_emb.npy`, `train_emb.npy`, `train.csv`
* Output: `hard_neg_pos_aware.csv` – top-K hard negative document IDs for each query

### 4. Create contrastive training data

```bash
python create_training_data.py
```

* Input:
  * `corpus.csv`
  * `train.csv`
  * `hard_neg_pos_aware.csv`
* Output:
  * `cleaned_corpus.csv` – cleaned, filtered corpus
  * `full_hard_neg.csv` – (query_id, pos_id, neg_id) rows for training

### 5. Train your embedding model

```bash
python train_embedding.py
```

* Reads the contrastive triplets from `full_hard_neg.csv`
* Trains a dual-encoder with in-batch + hard-negative contrastive loss
* Saves a new checkpoint under `Contrastive_E5/checkpoints/`

---

## 🧪 Baseline Notebooks

* **BM25.ipynb** – full implementation, evaluation and inference example
* **BM25F.ipynb** – similar to BM25 but with field-based weighting
* **FAISS.ipynb** – build a FAISS index on `full_emb.npy`, run nearest-neighbor retrieval

Each notebook is self-contained: load your cleaned data, run retrieval, compute accuracy metrics.

---

## ⚙️ Preprocessing

Run `Preprocess.ipynb` to normalize text, remove noise, tokenize and POS-tag with Underthesea.
This produces `corpus.csv` and `train.csv` ready for embedding and mining.

---

## 🛠️ Requirements

All Python dependencies live in:
```
Contrastive_E5/requirements.txt
```

Install via:
```bash
pip install -r Contrastive_E5/requirements.txt
```

You'll also need the usual ML stack:
* Python ≥ 3.8
* PyTorch
* HuggingFace Transformers
* FAISS
* Underthesea (for Vietnamese tokenization)

---

## 📄 License

This project is licensed under MIT. Feel free to adapt the pipeline for your own legal retrieval tasks!

---

*Happy retrieving!*