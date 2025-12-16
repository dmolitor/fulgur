# fulgur


<!-- badges: start -->

<!-- badges: end -->

The goal of fulgur is to facilitate estimating linear models on
extremely large (e.g. out-of-core) datasets. fulgur provides a simple
but flexible interface, allowing model specification via formula syntax
(see [formulaic](https://github.com/matthewwardrop/formulaic) for
details), and can fit a variety of linear models including OLS, Ridge,
Lasso, Elastic Net, etc.

fulgur is built on top of [scikit-learn’s Stochastic Gradient Descent
(SGD)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
learners, and inherits all the corresponding parameters and attributes.
Below we demonstrate a simple regression and classification problem on a
large dataset of airline arrivals.

## Regression - airline arrival delays

Below we’ll demonstrate fitting an OLS regression model to predict
airline arrival delay time using a large publicly available dataset of
US airline on-time performance.

First we’ll load the necessary packages.

``` python
from fulgur.classification import LargeLinearClassifier
from fulgur.regression import LargeLinearRegressor
import polars as pl
```

### Load data

We’ll load the airlines dataset from AWS as a [Polars lazy
DataFrame](https://docs.pola.rs/user-guide/lazy/using/). This dataset
has approximately 120 million rows which can be too large to fit into
memory on some computers, particularly when fitting regression models.
We will then split the data into a training and evaluation set for model
assessment.

``` python
airline = pl.scan_parquet(
    "s3://fulgur-large-regression/airline/",
    storage_options={"skip_signature": "true"}
)
airline = airline.drop("index").with_row_index(offset=1)

# Get train set and an evaluation set
airline_train = airline.head(118000000)
airline_eval = airline.tail(914458)
```

### Query the data

Next we’ll design a query to drop rows in our data with missing values
and to filter out cancelled flights.

``` python
def query_fn(x: pl.LazyFrame):
    return (
        x
        .filter(pl.col("cancelled").ne(1))
        .drop_nulls(["arrival_delay", "departure_delay", "scheduled_elapsed_time"])
        .drop_nans(["arrival_delay", "departure_delay", "scheduled_elapsed_time"])
    )
```

### Fit the model

Finally, we’ll train our OLS model. We will predict flights’ arrival
delays using departure delays and scheduled elapsed time as predictors.

``` python
llm = LargeLinearRegressor(
    formula="arrival_delay ~ departure_delay + scheduled_elapsed_time",
    data=airline_train,
    query=query_fn,
    batch_size=10000,
    type="ols",
    learning_rate="invscaling"
)
llm.fit(verbose=False)
```

### Compare to standard OLS

We’ll compare fulgur’s SGD-based coefficients and Root Mean Squared
Error (RMSE) on the evaluation set to those from a standard OLS model.

<details class="code-fold">
<summary>Code</summary>

``` python
from fulgur.utils import encode_categorical, scale_numeric
from sklearn.linear_model import LinearRegression

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5

# Construct design matrix
data = llm.query(airline_train) if llm.query else airline_train
stats = llm.stats
data = scale_numeric(data=data, stats=stats)
data = encode_categorical(data=data, formula=llm.formula)
comparison_data = llm.prep(data.collect(), output="numpy")
X = comparison_data.rhs
y = comparison_data.lhs.ravel()

# Fit standard OLS
ols_model = LinearRegression(fit_intercept=False)
ols_model.fit(X, y)

print(f"SGD Coefficients: {[round(float(x), 3) for x in llm.model.coef_]}")
print(f"OLS Coefficients: {[round(float(x), 3) for x in ols_model.coef_]}")

# Compare SGD RMSE to OLS RMSE
pred_data = llm.query(airline_eval) if llm.query else airline_eval
pred_data = scale_numeric(data=pred_data, stats=stats)
pred_data = encode_categorical(data=pred_data, formula=llm.formula)
pred_data = llm.prep(pred_data.collect(), output="numpy")
X_pred = pred_data.rhs
sgd_preds = llm.predict(airline_eval)
ols_preds = ols_model.predict(X_pred)

y_truth = pred_data.lhs.ravel()
sgd_rmse = rmse(y_truth, sgd_preds)
ols_rmse = rmse(y_truth, ols_preds)
print("-----------------------------------------")
print(f"SGD RMSE: {round(float(sgd_rmse), 3)}")
print(f"OLS RMSE: {round(float(ols_rmse), 3)}")
```

</details>

    SGD Coefficients: [8.823, 28.646, -1.694]
    OLS Coefficients: [7.034, 26.378, -0.744]
    -----------------------------------------
    SGD RMSE: 14.18
    OLS RMSE: 14.284

We see that our fulgur SGD-based OLS model achieves similar (slightly
better) hold-out error to the standard OLS model while being far more
memory and computationally efficient.

## Classification - airline arrival delays

Next, we’ll use the same dataset to demonstrate fitting a classification
model (logistic regression) to predict whether or not a flight will be
delayed. In short, we take the regression problem above and binarize the
outcome variable, coding flights as delayed (1) or not delayed (0).

``` python
def query_fn(x: pl.LazyFrame):
    return (
        x
        .filter(pl.col("cancelled").ne(1))
        .drop_nulls(["arrival_delay", "departure_delay", "scheduled_elapsed_time"])
        .drop_nans(["arrival_delay", "departure_delay", "scheduled_elapsed_time"])
        .with_columns(pl.col("arrival_delay").gt(0).cast(int).alias("is_delayed"))
    )
```

### Fit the model

Next, we’ll train our logistic regression model. We will predict whether
flights will be delayed using their departure delays and scheduled
elapsed time as predictors.

``` python
llm_cf = LargeLinearClassifier(
    formula="is_delayed ~ departure_delay + scheduled_elapsed_time",
    data=airline_train,
    query=query_fn,
    batch_size=10000,
    type="logistic",
    learning_rate="invscaling"
)
llm_cf.fit(verbose=False)
```

### Compare to standard logistic regression

As before, we’ll compare fulgur’s SGD-based coefficients and accuracy on
the evaluation set to those from a standard logistic regression model.

<details class="code-fold">
<summary>Code</summary>

``` python
from fulgur.utils import encode_categorical, scale_numeric
import numpy as np
from sklearn.linear_model import LogisticRegression

def accuracy(y_true, y_pred):
    return float((y_true == y_pred).sum()/len(y_true))

# Construct design matrix
data = llm_cf.query(airline_train) if llm_cf.query else airline_train
stats = llm_cf.stats
data = scale_numeric(data=data, stats=stats)
data = encode_categorical(data=data, formula=llm_cf.formula)
comparison_data = llm_cf.prep(data.collect(), output="numpy")
X = comparison_data.rhs
y = comparison_data.lhs.ravel()

# Fit standard logistic regression
lr_model = LogisticRegression(C=np.inf, fit_intercept=False)
lr_model.fit(X, y)

print(f"SGD Coefficients: {[round(float(x), 3) for x in llm_cf.model.coef_.ravel()]}")
print(f"Logit Coefficients: {[round(float(x), 3) for x in lr_model.coef_.ravel()]}")

# Compare SGD Accuracy to Logit Accuracy
pred_data = llm_cf.query(airline_eval) if llm_cf.query else airline_eval
pred_data = scale_numeric(data=pred_data, stats=stats)
pred_data = encode_categorical(data=pred_data, formula=llm_cf.formula)
pred_data = llm_cf.prep(pred_data.collect(), output="numpy")
X_pred = pred_data.rhs
sgd_preds = llm_cf.predict(airline_eval)
logit_preds = lr_model.predict(X_pred)

y_truth = pred_data.lhs.ravel()
sgd_acc = accuracy(y_truth, sgd_preds)
logit_acc = accuracy(y_truth, logit_preds)
print("-----------------------------------------")
print(f"Naive Accuracy: {round(1 - y_truth.mean(), 3)}")
print(f"SGD Accuracy: {round(float(sgd_acc), 3)}")
print(f"Logit Accuracy: {round(float(logit_acc), 3)}")
```

</details>

    SGD Coefficients: [0.534, 3.818, 0.01]
    Logit Coefficients: [0.869, 4.82, -0.033]
    -----------------------------------------
    Naive Accuracy: 0.543
    SGD Accuracy: 0.774
    Logit Accuracy: 0.775
