# Event Preference Model Benchmark (Binary)

This notebook benchmarks multiple ML classifiers to predict a user’s **Event_Preference** from survey-style inputs.
Target is **binary**:

- `Educational` → `0`
- `Entertainment` → `1` (used as the “positive” class for ROC/PR metrics)

---

## What’s inside

### 1) Stratified train/test split
Keeps the class ratio similar in train and test:
- `train_test_split(..., stratify=y)`

### 2) Model benchmark + tuning (GridSearchCV)
For each model we run a small hyperparameter search (`cv=5`, scoring=`accuracy`):
- Logistic Regression
- Random Forest
- SVM (with `probability=True`)
- Ridge Classifier
- Decision Tree
- Gradient Boosting
- XGBoost 

### 3) Evaluation per model
- Classification report + confusion matrix
- Accuracy
- Balanced Accuracy
- MCC
- ROC-AUC + PR-AUC (Average Precision), when score/probability is available
- Simple **threshold tuning** for probability models (`0.3 / 0.5 / 0.7`)

---

## Why these models?

We benchmark a **mix of linear + tree-based + kernel** models because survey-style/tabular data can behave very differently depending on:
- linear separability vs. complex interactions
- how much noise/outliers you have
- whether you need well-calibrated probabilities (thresholding)
- dataset size (some models scale better than others)

### Logistic Regression (strong linear baseline)
**Why it’s here:**
- Standard **first serious baseline** for binary classification on tabular data.
- Fast, stable, and often surprisingly competitive.
- Provides probabilities (`predict_proba`) → useful for threshold tuning.

**What it’s good at:**
- roughly linear relationships
- interpretability (coefficients)

**Key hyperparam tuned:**
- `C` = regularization strength (smaller = more regularization)

### Ridge Classifier (linear, robust, score-based)
**Why it’s here:**
- Another strong linear option with **L2 regularization**, often very robust.
- Uses `decision_function` (score), not probabilities by default → good for ROC-AUC, but probability thresholding may require calibration.

**What it’s good at:**
- stable linear classification
- high-dimensional feature spaces

**Key hyperparam tuned:**
- `alpha` = regularization strength

### SVM (Support Vector Machine) — linear + RBF kernel
**Why it’s here:**
- Classic “power model” when boundaries are tricky.
- The **RBF kernel** can capture non-linear decision boundaries.
- We set `probability=True` so we can do threshold tuning (note: slower due to internal calibration).

**What it’s good at:**
- medium-sized datasets
- complex class boundaries without explicit feature interaction engineering

**Key hyperparams tuned:**
- `C` controls margin vs misclassification tradeoff
- `kernel` chooses linear vs non-linear boundary

### Decision Tree (interpretable non-linear baseline)
**Why it’s here:**
- Transparent, easy-to-explain non-linear model.
- Great sanity check to see if **simple rule splits** already work.

**What it’s good at:**
- capturing basic feature interactions
- interpretability (you can visualize the tree)

**Risk:**
- easily overfits → hence tuning `max_depth`

### Random Forest (robust non-linear ensemble)
**Why it’s here:**
- Strong general-purpose model for tabular data.
- Reduces overfitting vs. a single tree by averaging many trees.
- Usually robust to noise and non-linearities.

**What it’s good at:**
- non-linear interactions
- stable performance with minimal feature engineering

**Key hyperparams tuned:**
- `n_estimators` number of trees
- `max_depth` controls complexity

### Gradient Boosting (boosted trees, strong tabular performer)
**Why it’s here:**
- Boosting often beats RandomForest on structured/tabular tasks.
- Builds trees sequentially to fix errors → strong at capturing subtle patterns.

**What it’s good at:**
- high performance on tabular data
- capturing interactions without manual feature engineering

**Key hyperparams tuned:**
- `n_estimators` + `learning_rate` control bias/variance tradeoff

### XGBoost (optional, “production-grade” boosted trees)
**Why it’s here:**
- Frequently **top-tier** on tabular binary classification.
- Strong regularization + speed compared to classic sklearn boosting.
- Good scores + usable probabilities (`predict_proba`) for threshold decisions.


**Key hyperparams tuned:**
- `n_estimators`, `learning_rate`, `max_depth`

---

## Why tuning + multiple metrics?

We tune each model with **5-fold CV** (`GridSearchCV`) because a single train/test split can be misleading.

We report multiple metrics because each catches different failure modes:
- **Accuracy**: can look “fine” even if the model ignores one class
- **Balanced Accuracy**: fairer when classes are uneven
- **MCC**: strong single-number score for binary classification (good even with imbalance)
- **ROC-AUC / PR-AUC**: measures ranking quality across thresholds (useful if you care about probabilities)

---

## Why threshold tuning?

Default threshold `0.5` is just a convention.
If the product goal changes, you often want a different tradeoff:
- higher recall for Entertainment (catch more Entertainment users)
- higher precision (label Entertainment only when you’re confident)

So we test `0.3 / 0.5 / 0.7` to quickly see how precision/recall/F1 move.

---

## How to use

### 1) Prepare your data
You need:
- `X_processed`: numeric feature matrix (already preprocessed/encoded)
- `y`: target array (0/1)

If you have raw survey columns, run your preprocessing pipeline first to create `X_processed`.

### 2) Run the benchmark cell
Execute the benchmark cell. It will:
- train + tune each model
- print metrics
- show confusion matrices
- store best models in:
  - `best_models[name]`
- store summary in:
  - `results[name]`

### 3) Pick and export the final model
After you decide which model to use, export it (together with preprocessing if you have a full pipeline):

```python
import cloudpickle

final_model = best_models["XGBoost"]  # example
with open("pipeline.pkl", "wb") as f:
    cloudpickle.dump(final_model, f)
```

If you have an end-to-end `Pipeline(preprocessor + model)`, export the full pipeline instead — that’s the cleanest for production.

---

## How we choose the model (in practice)

We don’t pick “by vibe”. We pick by:

1. **Primary metric**
   - If you care about “ranking quality” (good probabilities / good ordering): **ROC-AUC** / **PR-AUC**
   - If you want one robust number and fair treatment of both classes: **MCC** / **Balanced Accuracy**
   - Accuracy is fine, but can hide issues when one class is easier.

2. **Stability**
   - We use **5-fold CV** in `GridSearchCV` to reduce “lucky split” effects.

3. **Confusion matrix sanity**
   - Even with good AUC, check whether the model is doing something stupid (e.g. predicting mostly one class).

4. **Threshold**
   - The “best model” might still need a different threshold than 0.5 (especially if you care more about recall or precision).

**Rule of thumb for this project:**
- Start with **AUC/AP** (ranking quality) + confirm with **MCC/Balanced Accuracy**.
- Then verify on confusion matrix + optionally pick a better threshold.

---

## What these metrics mean (short)

- **Accuracy**: overall correctness
- **Balanced Accuracy**: average recall for both classes (more fair)
- **MCC**: strong single-number score for binary classification (good even with imbalance)
- **ROC-AUC**: how well the model ranks Entertainment above Educational across thresholds
- **PR-AUC (AP)**: like ROC-AUC but more sensitive to positive-class quality

> ROC/PR metrics require a “positive class” by definition.  
> Here: **Entertainment = 1** is just a convention; both classes are equally important.

---

## Real-world use: microservice scenario

This fits a practical microservice like:

**`POST /recommendation/event-preference`**

User fills a short survey (age, city size, hobbies, personality type, etc.).
The service returns:
- predicted class: Educational / Entertainment
- probability score (optional)
- (optionally) explanation/feature importance if the model supports it

### Example flow
1. Frontend shows a survey form
2. Backend microservice:
   - loads `pipeline.pkl`
   - transforms the input
   - returns prediction

This is exactly why we benchmark models:  
we want a model that is **accurate**, **stable**, and (ideally) produces **useful probabilities** for a real app.
