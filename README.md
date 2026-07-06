# Decision Tree & Random Forest Classification Lab

A hands-on lab covering decision tree and random forest classifiers using
scikit-learn, applied to two datasets: the classic Iris dataset and a
real-world loan repayment dataset.

## Contents

### Part A & B — Iris Classification
- Trained a `DecisionTreeClassifier` and a `RandomForestClassifier` on the
  Iris dataset (via `sklearn.datasets.load_iris`)
- Predicted the class and class probabilities for a new observation
- Evaluated both models with classification reports
- Visualised the decision tree structure using `graphviz` / `pydotplus`

### Part C — Loan Repayment Prediction
- Applied the same two models to a loan repayment dataset (LendingClub-style
  data, `not.fully.paid` as target)
- Compared Decision Tree vs. Random Forest performance via classification
  reports and confusion matrices
- Discussed class imbalance: both models struggled to identify the minority
  class (loans not fully paid), with Random Forest showing higher overall
  accuracy but poor recall on that class
- Included written analysis on model comparison and where feature
  engineering could improve results

### Bonus — Linear Regression
- Quick demonstration of `LinearRegression` on a small synthetic dataset

## Repository Structure

```
├── decision_tree_random_forest_lab.ipynb   # Main notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Data

- **Iris dataset:** loaded directly from `sklearn.datasets`, no download needed
- **Loan dataset:** referenced in the original notebook via a Google Colab
  path (`/content/drive/MyDrive/...`). It is not included here — to run
  Part C locally, source a loan repayment dataset with a `not.fully.paid`
  target column (e.g. the LendingClub loan data used in the "Introduction
  to Data Science" course materials) and update the file path in the
  notebook to point to your local copy.

## Setup

```bash
git clone https://github.com/<your-username>/decision-tree-random-forest-lab.git
cd decision-tree-random-forest-lab
pip install -r requirements.txt
```

Tree visualisation also requires the Graphviz system package:

```bash
# macOS
brew install graphviz
# Ubuntu/Debian
sudo apt-get install graphviz
```

Then launch:

```bash
jupyter notebook decision_tree_random_forest_lab.ipynb
```

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Graphviz
