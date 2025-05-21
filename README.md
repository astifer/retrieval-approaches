# Retrieval Approaches for Question-Answering

This project implements various retrieval approaches for question-answering using the Natural Questions dataset.

## Project Structure

- `metrics.py` - Implementation of retrieval metrics (Recall@K, MRR)
- `tfidf_baseline.py` - TF-IDF baseline implementation
- `e5_baseline.py` - E5 model baseline implementation
- `e5_train.py` - E5 model training with Contrastive and Triplet Loss
- `e5_hard_negatives.py` - E5 model training with hard negatives
- `utils.py` - Utility functions for data processing and evaluation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run experiments:
```bash
# Run on GPU (default if available)
python tfidf_baseline.py  # Task 2
python e5_baseline.py     # Task 3
python e5_train.py        # Task 4
python e5_hard_negatives.py  # Task 5

# Run on CPU
python e5_baseline.py --device cpu
python e5_train.py --device cpu
python e5_hard_negatives.py --device cpu
```

## Results

# tfidf_baseline
|metric| value|
| ---- | ------|
|Recall@1| 0.3555|
|Recall@3 |0.5707|
|Recall@10 |0.7535|
|MRR | 0.4828|

# e5 baseline
|metric| value|
| ---- | ------|
|Recall@1| 0.6868|
|Recall@3 |0.8837|
|Recall@10 |0.9655|
|MRR | 0.7916|
