# 02456 Retrieval Project

Starter code for information retrieval project work in the 02456 Deep Learning course (Fall 2026).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### Download the tokenizer

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"
```

## Usage

For replicating the results in the report, run the results.ipynb notebook. 
TF-IDF and BM-25 is commented out in the main.py such that if you run the notebook, which call main.py sepperatly, the you will not compute TF-IDF and BM-25 twice. 

If you would like to run TF-IDF, BM-25, and DPR though the main.py, then uncomment line 36-69. 


```bash
python main.py            
python main.py top_k=20   # Override a config value
```

This prints recall@k for the train, validation and test splits. Settings are managed with Hydra in `config/`.

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval using TF-IDF and a preliminary evaluation pipeline.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall metric.
```
