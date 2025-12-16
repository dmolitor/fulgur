import formulaic as fml
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from typing import Callable

from fulgur.call_py import call_py, stream_data
from fulgur.model import FulgurModel
from fulgur.utils import (
    encode_categorical,
    lhs,
    scale_numeric,
    sgd_config_classification,
    summary_stats,
    unique,
)


class LargeLinearClassifier(BaseEstimator, FulgurModel):

    def __init__(
        self,
        formula: str,
        data: pl.LazyFrame,
        query: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
        batch_size: int = 1000,
        type: str = "logistic",
        learning_rate: str = "invscaling",
        eta0: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self._fitted = False
        self.batch_size = batch_size
        self.formula = fml.Formula(formula)
        loss, penalty = sgd_config_classification(type)
        if "fit_intercept" in kwargs:
            del kwargs["fit_intercept"]
        if "loss" in kwargs:
            loss = kwargs["loss"]
            del kwargs["loss"]
        if "penalty" in kwargs:
            penalty = kwargs["penalty"]
            del kwargs["penalty"]
        if "eta0" in kwargs:
            penalty = kwargs["eta0"]
            del kwargs["eta0"]
        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            learning_rate=learning_rate,
            fit_intercept=False,
            eta0=eta0,
            **kwargs,
        )
        self.query = query

        # Append necessary queries prior to model fitting
        data = query(data) if query else data

        # Calculate necessary summary stats for feature transformation
        self.stats = summary_stats(data, formula)
        data = scale_numeric(data=data, stats=self.stats)
        data = encode_categorical(data=data, formula=formula)

        # Get unique values for the response variable
        self._classes = unique(data, col=lhs(self.formula))

        # Store data for model fitting
        self.data = data

    def fit(self, verbose: bool = True):
        def fitting_fn(data: pl.DataFrame):
            prepped = self.prep(data, output="sparse")
            X = prepped.rhs
            y = prepped.lhs.toarray().ravel()
            self.model.partial_fit(X=X, y=y, classes=self._classes)
            return self.model

        fitted_model = call_py(
            stream_data,
            data=self.data,
            fn=fitting_fn,
            batch_size=self.batch_size,
            last=True,
            verbose=verbose,
        )
        self.model = fitted_model
        self._fitted = True
