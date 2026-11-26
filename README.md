# Ensemble Machine Learning Model For Cassiterite Provenance Classification 

This repository contains the dataset, training script, and serialized models used to classify cassiterite samples into two provenance groups from trace-element compositions and elemental ratios.

- **Group A:** Cassiterite from granite, pegmatite, greisen, and porphyry.
- **Group B:** Cassiterite from skarn, vein, and sulfide-ore.

The workflow trains and compares Random Forest, XGBoost, TabNet, and a final LightGBM model referred to as the `ensemble model` in the repository files.

## Repository contents

| File | Description |
|---|---|
| `Cst_data.xlsx` | Input dataset used for model training and evaluation |
| `Ensemble Model.py` | Complete preprocessing, visualization, model optimization, training, evaluation, and SHAP workflow |
| `best_rf_model.pkl` | Saved optimized Random Forest model |
| `best_xgb_model.pkl` | Saved optimized XGBoost model |
| `best_tabnet_model.pkl` | Saved optimized TabNet model |
| `best_ensem_model.pkl` | Saved final LightGBM model labelled as the ensemble model |
| `requirements.txt` | Core Python dependencies and versions used to serialize the models |
| `CITATION.cff` | Citation metadata |

## Dataset

`Cst_data.xlsx` contains **4,097 cassiterite analyses** and 22 columns. There are 1,037 Group A samples and 3,060 Group B samples.

| Sub-group | Number of samples |
|---|---:|
| A1 | 245 |
| A2 | 470 |
| A3 | 309 |
| A4 | 13 |
| B1 | 555 |
| B2 | 2,070 |
| B3 | 435 |

Three columns provide sample metadata and labels:

- `Sample ID`: sample identifier
- `Group`: binary target label (`A` or `B`)
- `Sub-group`: sub-group label used for UMAP visualization, not for model training

The following **19 numerical predictors** are used for classification:

`Al`, `Sc`, `Ti`, `V`, `Fe`, `Ga`, `W`, `Sb`, `Zr`, `Hf`, `Nb`, `Ta`, `U`, `Zr/Hf`, `Nb/Ta`, `Sb/W`, `Fe/Al`, `U/Hf`, and `U/Zr`.

The values in the supplied spreadsheet are already processed numerical inputs. The training script does not perform additional imputation or scaling, and the dataset contains no missing values.

## Training workflow

### 1. Load and prepare the data

The script reads `Cst_data.xlsx`, removes `Sample ID`, `Group`, and `Sub-group` from the predictor matrix, and encodes the `Group` labels with `LabelEncoder`. With the current data, the encoded classes are ordered as `A = 0` and `B = 1`.

### 2. Explore sub-group structure with UMAP

UMAP is applied to the 19 predictors using:

```python
umap.UMAP(
    n_components=2,
    n_neighbors=100,
    min_dist=0.5,
    random_state=2,
)
```

The resulting plot displays A1-A4 and B1-B3. These sub-group labels are used only for visualization.

### 3. Create training and test sets

The data are divided into 70% training and 30% test sets with stratification by Group:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=2025,
    stratify=y,
)
```

Stratification preserves the A:B class proportions in both subsets. The fixed random state makes the split repeatable within a compatible software and hardware environment.

### 4. Train the base models

#### Random Forest

Random Forest hyperparameters are optimized with `BayesSearchCV` using 50 search iterations and five-fold cross-validation. The optimization score is weighted one-vs-rest ROC AUC. The search covers the number of trees, tree depth, split and leaf sizes, feature sampling, and bootstrap sampling.

#### XGBoost

XGBoost is also optimized with `BayesSearchCV` using 50 search iterations, five-fold cross-validation, and weighted one-vs-rest ROC AUC. The search includes tree complexity, learning rate, row and column sampling, minimum child weight, gamma, and L1/L2 regularization.

#### TabNet

TabNet hyperparameters are optimized in 20 Optuna trials using a TPE sampler and five-fold stratified cross-validation. The search includes attention dimensions, decision steps, gamma, sparsity regularization, learning rate, and mask type. The final model uses early stopping with a maximum of 50 epochs.

The principal random seed is `2025` for data splitting and model optimization. UMAP uses `random_state=2`.

### 5. Train the final model labelled as the ensemble model

After training and evaluating the three base models, the script runs 50 Optuna trials to optimize an `LGBMClassifier`. The search maximizes test-set accuracy and tunes:

- number of estimators;
- learning rate;
- maximum tree depth;
- number of leaves;
- row subsampling; and
- column subsampling.

The optimized LightGBM model is trained on `X_train`, evaluated on `X_test`, and serialized as the final model.

> **Implementation note:** In the current `Ensemble Model.py`, this final model is a standalone LightGBM classifier trained directly on the original 19 predictors. Although the script output and model filename refer to it as a stacking or ensemble model, predictions from Random Forest, XGBoost, and TabNet are not passed to LightGBM, and the imported `StackingClassifier` is not instantiated. A true stacking implementation would train a meta-learner on out-of-fold predictions from the base models.

### 6. Evaluate and interpret the models

The script compares all models using:

- accuracy;
- weighted precision;
- weighted recall;
- weighted F1 score;
- ROC AUC; and
- confusion matrices.

It also creates learning curves and uses SHAP to examine feature contributions. TreeExplainer is used for Random Forest and XGBoost, while KernelExplainer is used for TabNet.

## Installation

Python 3.11 is recommended because the saved models were produced with the versions recorded in `requirements.txt`.

```bash
git clone https://github.com/jinghuawu/Cst-Submission-2025.git
cd Cst-Submission-2025
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the core dependencies and the additional packages imported by the full training script:

```bash
pip install -r requirements.txt
pip install optuna scikit-optimize shap umap-learn seaborn matplotlib scipy
```

Using different versions of Python, scikit-learn, XGBoost, LightGBM, PyTorch, or PyTorch TabNet may prevent the `.pkl` files from loading correctly or may produce slightly different results.

## Run the training script

Before running the script, change the two Windows paths near the beginning of `Ensemble Model.py`:

```python
out_dir = r'D:\Data'
data_path = r'D:\Data\Cst_data.xlsx'
```

For example, to run from the repository directory:

```python
out_dir = r'outputs'
data_path = r'Cst_data.xlsx'
```

Then execute:

```bash
python "Ensemble Model.py"
```

The complete workflow performs Bayesian/Optuna optimization and SHAP analysis, so it may require considerable CPU time. TabNet can use a compatible GPU through PyTorch when available.

## Use a saved model

The input column order must match the order used during training:

```python
import joblib
import pandas as pd

FEATURES = [
    'Al', 'Sc', 'Ti', 'V', 'Fe', 'Ga', 'W', 'Sb', 'Zr', 'Hf',
    'Nb', 'Ta', 'U', 'Zr/Hf', 'Nb/Ta', 'Sb/W', 'Fe/Al', 'U/Hf', 'U/Zr'
]

data = pd.read_excel('new_cassiterite_data.xlsx')
X_new = data[FEATURES].to_numpy()

model = joblib.load('best_ensem_model.pkl')
encoded_predictions = model.predict(X_new).astype(int)
probabilities = model.predict_proba(X_new)

label_map = {0: 'A', 1: 'B'}
predicted_groups = [label_map[value] for value in encoded_predictions]

result = data.copy()
result['Predicted Group'] = predicted_groups
result['Probability A'] = probabilities[:, 0]
result['Probability B'] = probabilities[:, 1]
result.to_excel('cassiterite_predictions.xlsx', index=False)
```

Only load `.pkl` files obtained from a trusted source, because Python pickle-based formats can execute arbitrary code during loading.

## Reproducibility notes

- Keep the supplied feature names and column order unchanged.
- Use the dependency versions in `requirements.txt` when loading the saved models.
- Seeds reduce stochastic variation but do not guarantee bitwise-identical results across different CPUs, GPUs, operating systems, thread counts, or package versions.
- The current LightGBM optimization evaluates every Optuna trial directly on the held-out test set. For stricter model assessment, optimize with cross-validation or a separate validation set and reserve the test set for one final evaluation.
- The final TabNet fit uses the test set as its validation set for early stopping. A separate validation subset is preferable when reporting unbiased test performance.

## Citation

If you use the dataset or models, please cite this repository using the metadata in `CITATION.cff` and cite the associated publication when available.

## License

No license file is currently included. Please contact the repository owner before redistributing or reusing the dataset, code, or trained models beyond the permissions provided by the associated publication.
