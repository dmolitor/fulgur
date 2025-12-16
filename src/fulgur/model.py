import polars as pl
from typing import Any, Callable

from fulgur.utils import encode_categorical, scale_numeric


class FulgurModel:

    def __init__(self):
        pass

    def prep(
        self, data: pl.DataFrame, output=["numpy", "sparse", "narwhals", "pandas"]
    ) -> Any:
        output = output[0] if isinstance(output, list) else output
        return self.formula.get_model_matrix(data, output=output, na_action="ignore")

    def predict(self, data: pl.LazyFrame | pl.DataFrame):
        if not self._fitted:
            raise ValueError("Model must be fitted before predictions can be made.")
        data = self.query(data) if self.query else data
        stats = self.stats
        data = scale_numeric(data=data, stats=stats)
        data = encode_categorical(data=data, formula=self.formula)
        if isinstance(data, pl.LazyFrame):
            data = self.prep(data.collect(engine="streaming"), output="sparse")
        elif isinstance(data, pl.DataFrame):
            data = self.prep(data, output="sparse")
        else:
            raise TypeError(
                "Internal error; data is neither a Polars LazyFrame nor DataFrame"
            )
        X = data.rhs
        return self.model.predict(X)

    def fit_with_error(
        self, data: pl.LazyFrame, error_fn: Callable, verbose: bool = True
    ):
        # TODO: right now a lot of code is duplicated from fit; at some point refactor
        formula = self.formula.lhs.required_variables
        if len(formula) != 1:
            raise ValueError("Malformed formula. LHS must be a single variable.")
        if self.query:
            y_truth = self.query(data).select(pl.col(formula.pop()))
        else:
            y_truth = data.select(pl.col(formula.pop()))
        y_truth = y_truth.collect(engine="streaming").to_series().to_numpy()

        def fitting_fn(d: pl.DataFrame):
            if not hasattr(self, "error"):
                self.error = list()
            # Prep data and partially fit
            prepped = self.prep(d, output="sparse")
            X = prepped.rhs
            y = prepped.lhs.toarray().ravel()
            self.model.partial_fit(X=X, y=y)
            self._fitted = True
            # Predict on hold-out data and calculate hold-out error
            y_pred = self.predict(data)
            error = error_fn(y_truth, y_pred)
            self.error.append(error)
            return (self.model, self.error)

        fitted_model = call_py(
            stream_data,
            data=self.data,
            fn=fitting_fn,
            batch_size=self.batch_size,
            last=True,
            verbose=verbose,
        )
        self.model, self.error = fitted_model
        self._fitted = True
