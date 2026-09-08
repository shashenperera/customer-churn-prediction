# Customer Churn Prediction: A Comparative Study of XGBoost and MLP Classifiers

A machine learning project comparing **XGBoost**, **Keras/TensorFlow MLP**, and **Scikit-learn MLPClassifier** for binary customer churn prediction.

The project covers data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter optimization, and evaluation using classification metrics, confusion matrices, and ROC-AUC.

------------------------------------------------------------------------

## Project Objectives

The main objectives of this project are to:

- Develop machine learning models for customer churn prediction.
- Compare the predictive performance of XGBoost and two MLP implementations.
- Investigate relationships between customer characteristics and churn.
- Apply appropriate preprocessing, categorical encoding, and feature scaling.
- Optimize model hyperparameters using 5-fold cross-validation and F1-score.
- Compare the final models using Accuracy, Precision, Recall, F1-score, confusion matrices, and ROC-AUC.
- Identify the best-performing model for this dataset.

------------------------------------------------------------------------

## Dataset

The project uses a customer churn dataset containing **5,000 customer records and 12 features**.

The variables include:

- Customer ID
- Age
- Gender
- Subscription Length
- Region
- Payment Method
- Support Tickets Raised
- Satisfaction Score
- Discount Offered
- Last Activity
- Monthly Spend
- Churned (binary target)

The target variable, **Churned**, represents whether a customer churned or was retained.

> **Dataset note:** See [`data/README.md`](data/README.md) for information about the dataset and redistribution considerations.

------------------------------------------------------------------------

## Technologies

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SciKeras
- Jupyter Notebook

------------------------------------------------------------------------

## Project Structure

``` text
customer-churn-prediction/
│
├── data/
│   └── README.md
│
├── figures/
│   ├── eda.png
│   ├── confusion_matrices.png
│   └── roc_curves.png
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

------------------------------------------------------------------------

## Methodology

``` text
Data Gathering
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering / Selection
      ↓
Categorical Encoding
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Hyperparameter Optimization
      ↓
Model Evaluation
      ↓
Model Comparison
```

### Data preprocessing

- Missing values were checked for `Age` and `Satisfaction_Score`.
- Both variables contained 500 missing observations (10% of the dataset).
- Distribution, skewness, and potential outliers were considered before imputation.
- The notebook uses mean/median selection based on skewness and outlier checks.
- `Customer_ID` was removed because it is an identifier rather than a meaningful predictive feature.
- Categorical variables were one-hot encoded using `pd.get_dummies(..., drop_first=True)`.
- The data was divided into training and testing sets using an **80/20 stratified split**.
- `StandardScaler` was used for the MLP models through pipelines.

------------------------------------------------------------------------

## Exploratory Data Analysis

The notebook investigates:

- Missing-value patterns
- Churn distribution
- Correlation with churn
- Multicollinearity among numerical variables
- Numerical feature distributions by churn status
- Churn rates across categorical variables

The analysis indicates that **Monthly Spend** and **Satisfaction Score** have negative relationships with churn, while **Support Tickets Raised** shows a positive relationship with churn.

Add exported notebook figures to the `figures/` directory and display them here:

### Correlation with Churn

![Correlation with Churn](figures/correlation_with_churn.png)

### Multicollinearity Check

![Correlation Matrix](figures/multicollinearity.png)

### Churn Distribution

![Churn Distribution](figures/churn_distribution.png)

------------------------------------------------------------------------

## Models

### 1. XGBoost Classifier

XGBoost was used as a gradient-boosted tree classifier.

Hyperparameter optimization was performed with `GridSearchCV` using:

- 5-fold cross-validation
- F1-score as the optimization metric
- `n_estimators`
- `max_depth`
- `learning_rate`

The search used:

``` text
n_estimators: 482, 483, 484
max_depth: 2, 3, 4
learning_rate: 0.095, 0.100, 0.105
```

------------------------------------------------------------------------

### 2. Keras / TensorFlow MLP

A neural network was implemented using TensorFlow/Keras.

The architecture contains:

- Two fully connected hidden layers
- ReLU activation
- Dropout regularization
- One sigmoid output neuron for binary classification
- Adam optimizer
- Binary cross-entropy loss

The hyperparameter search considered:

``` text
neurons: 200, 220, 240
dropout rate: 0.3, 0.4
learning rate: 0.005, 0.007
epochs: 15
batch size: 56
```

`StandardScaler` was applied within a pipeline.

------------------------------------------------------------------------

### 3. Scikit-learn MLP

The third model uses `MLPClassifier` from Scikit-learn.

Hyperparameter optimization considered:

``` text
hidden layer size: 109, 110, 111
alpha: 0.085, 0.090, 0.095
learning rate initialization: 0.012, 0.013, 0.014
```

The model uses the Adam solver and `max_iter=500`, with scaling performed through a pipeline.

------------------------------------------------------------------------

## Hyperparameter Optimization

All three models were optimized using:

- `GridSearchCV`
- 5-fold cross-validation
- F1-score as the scoring metric

F1-score was selected because it provides a balance between precision and recall for the binary churn classification problem.

------------------------------------------------------------------------

## Results

### Test Performance

| Model            |  Accuracy | Precision |     Recall |  F1-Score |
|------------------|----------:|----------:|-----------:|----------:|
| **XGBoost**      | **99.7%** |     99.3% | **100.0%** | **99.7%** |
| Scikit-learn MLP |     98.9% | **99.5%** |      98.0% |     98.8% |
| Keras MLP        |     98.1% |     96.7% |      99.1% |     97.9% |

**XGBoost achieved the best overall test performance**, with an F1-score of 99.7% and perfect recall on the test set.

### ROC-AUC

| Model            |    ROC-AUC |
|------------------|-----------:|
| **XGBoost**      | **1.0000** |
| Scikit-learn MLP |     0.9995 |
| Keras MLP        |     0.9819 |

The ROC-AUC results show excellent discriminative performance for all three models, with XGBoost achieving the highest AUC.

------------------------------------------------------------------------

## Confusion Matrices

Confusion matrices were used to evaluate the classification performance of each model by examining true positives, true negatives, false positives, and false negatives.

| Keras MLP | Scikit-learn MLP | XGBoost |
|-----------------------|---------------------------|----------------------|
| ![](figures/cm_keras.png) | ![](figures/cm_mlp_sklearn.png) | ![](figures/cm_xgb.png) |

------------------------------------------------------------------------

## ROC Curve Comparison

![ROC Curves](figures/roc_curves.png)

The ROC curves compare the ability of the three models to distinguish between churned and retained customers.

------------------------------------------------------------------------

## Key Findings

1.  **XGBoost was the strongest overall model**, achieving the highest Accuracy and F1-score.
2.  XGBoost achieved **100% recall**, meaning all churned customers in the test set were identified by the final classifier.
3.  **Scikit-learn MLP** also performed extremely well and achieved the highest precision among the three models.
4.  **Keras MLP** produced strong results but performed below the other two models on the reported test metrics.
5.  All three models achieved very high ROC-AUC values, indicating excellent class-discrimination ability on this dataset.
6.  The comparison demonstrates that a simpler Scikit-learn MLP can be a strong alternative to a custom Keras neural network for this dataset.

------------------------------------------------------------------------

## Reproducibility

Create and activate a Python environment, then install the required packages:

``` bash
pip install -r requirements.txt
```

Place the dataset at:

``` text
data/churn.csv
```

The notebook currently loads the dataset using:

``` python
pd.read_csv("../data/churn.csv")
```

Therefore, if the notebook is moved into a `notebooks/` directory, the existing relative path is appropriate.

Then open the notebook with Jupyter:

``` bash
jupyter notebook
```

or:

``` bash
jupyter lab
```

------------------------------------------------------------------------

## Evaluation Metrics

The project reports:

- **Accuracy** — proportion of correctly classified customers.
- **Precision** — proportion of predicted churners who actually churned.
- **Recall** — proportion of actual churners correctly identified.
- **F1-score** — harmonic mean of precision and recall.
- **Confusion matrix** — summarizes correct and incorrect classifications.
- **ROC-AUC** — measures the model's ability to discriminate between the two classes across classification thresholds.

------------------------------------------------------------------------

## Limitations

The reported results are based on a single train/test split and the available dataset. Very high test performance should therefore be interpreted in the context of this particular dataset and preprocessing pipeline.

The repository does not claim that the reported performance will generalise to every customer population or future dataset.

------------------------------------------------------------------------

## Future Improvements

Possible extensions include:

- Feature engineering and feature selection.
- Ensemble methods combining multiple classifiers.
- More extensive cross-validation and repeated evaluation.
- Probability calibration and threshold optimisation.
- Model interpretability using SHAP or related methods.
- Testing the models on an independent external dataset.
- Experimenting with additional neural-network architectures.

------------------------------------------------------------------------

## Author

**Shashen Perera**

Project completed as a machine learning study comparing tree-based and neural-network classifiers for customer churn prediction.
