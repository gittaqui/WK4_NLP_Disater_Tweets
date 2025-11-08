# NLP with Disaster Tweets

This repository hosts a baseline workflow for the Kaggle competition ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started). The goal is to classify whether a tweet describes a real disaster (label `1`) or not (label `0`).

## Project Structure

- `data/` contains the Kaggle-provided CSV files (`train.csv`, `test.csv`, `sample_submission.csv`) and the generated `submission.csv` file.
- `notebooks/disaster_tweets_workflow.ipynb` is the primary analysis notebook that loads the data, engineers features, trains a model, and exports predictions.

## Notebook Walkthrough

The notebook is organised as nine executable cells, each preceded by a short markdown description inside the notebook:

1. Imports core Python and scikit-learn utilities used throughout the workflow.
2. Loads the train, test, and sample submission files from `data/`, then previews labelled tweets.
3. Inspects the class distribution of the target label to understand baseline imbalance.
4. Reports column-wise missing value ratios to identify any cleanup requirements.
5. Normalises tweet text by lowercasing, stripping URLs, and removing non-alphanumeric characters.
6. Computes a simple character-length feature to compare tweet length across classes.
7. Builds a TF-IDF + logistic regression pipeline and evaluates it with 5-fold stratified cross-validation.
8. Fits the final pipeline on the full training set, predicts the test labels, and writes `submission.csv`.

## Key Outputs

- **Cross-validation performance:**
  - Per-fold F1 scores: `[0.75899844, 0.75910147, 0.72882673, 0.75, 0.75586854]`
  - Mean F1 score: `0.7506 ± 0.0114`
- **Submission artifact:** The final cell saves predictions to `data/submission.csv` and previews the first few rows:
  | id | target |
  | -- | ------ |
  | 0  | 1      |
  | 2  | 1      |
  | 3  | 1      |
  | 9  | 1      |
  | 11 | 1      |

## Usage

1. Place the Kaggle competition CSV files (`train.csv`, `test.csv`, `sample_submission.csv`) in the `data/` directory.
2. Open `notebooks/disaster_tweets_workflow.ipynb` in Jupyter or VS Code and run the cells sequentially to reproduce the results and regenerate `submission.csv`.
3. Submit `data/submission.csv` to Kaggle to receive a leaderboard score.
