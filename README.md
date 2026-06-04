# Spam Detection Public Data

Notebook and helper-code workspace for experimenting with text classification over public spam-detection data.

## What is here

- `doc2vec.ipynb` - exploratory Doc2Vec modeling notebook.
- `Tf_IDF+RF.ipynb` - exploratory TF-IDF plus Random Forest notebook.
- `pd_doc2vec.py` - reusable Doc2Vec-style helper class.
- `train.csv`, `test.csv`, `sample_submission.csv` - data files used by the notebooks.

## How to inspect

Open the notebooks in Jupyter, VS Code, or another notebook viewer. The helper script uses older versions of Gensim/scikit-learn APIs, so expect dependency updates before rerunning end to end.

## Reproducible demo

Run a deterministic notebook-free baseline over the first 2,500 training rows:

```sh
python3 scripts/demo.py
```

The demo trains a small standard-library Naive Bayes classifier, prints
validation metrics, and writes `outputs/demo_summary.md`. It is a smoke-test
baseline, not a claim that the original notebook models were fully retrained.

Lightweight syntax check for the reusable helper:

```sh
python -m compileall pd_doc2vec.py
```

## Caveats

- This is an exploratory ML/data artifact, not a packaged library.
- No pinned environment file is currently provided.
- Reproducibility work should start by adding a small requirements file and rerunning the notebooks with current library versions.
