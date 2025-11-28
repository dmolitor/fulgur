import formulaic as fml
from pathlib import Path
import polars as pl
from sklearn.linear_model import SGDClassifier, SGDRegressor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Any, Callable

from fulgur.call_py import call_py, stream_data
from fulgur.utils import (
    encode_categorical,
    scale_numeric,
    sgd_config_regression,
    summary_stats
)

class LargeLinearRegressor:

    def __init__(
        self,
        formula: str,
        data: pl.LazyFrame,
        query: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
        batch_size: int = 1000,
        type: str = "ols",
        learning_rate: str = "adaptive",
        **kwargs
    ):
        self.batch_size = batch_size
        self.formula = fml.Formula(formula)
        loss, penalty = sgd_config_regression(type)
        if "fit_intercept" in kwargs:
            del kwargs["fit_intercept"]
        if "loss" in kwargs:
            loss = kwargs["loss"]
            del kwargs["loss"]
        if "penalty" in kwargs:
            penalty = kwargs["penalty"]
            del kwargs["penalty"]
        self.model = SGDRegressor(loss=loss, penalty=penalty, learning_rate=learning_rate, fit_intercept=False, **kwargs)
        self.query = query
        self.stats = summary_stats(data, formula)

        # Append necessary queries prior to model fitting
        data = query(data) if query else data
        data = scale_numeric(data=data, stats=self.stats)
        data = encode_categorical(data=data, formula=formula)
        self.data = data
    
    def prep(self, data: pl.DataFrame, output=["numpy", "sparse", "narwhals", "pandas"]) -> Any:
        output = output[0] if isinstance(output, list) else output
        return self.formula.get_model_matrix(data, output=output)
    
    def fit(self, verbose: bool = True):
        def fitting_fn(data: pl.DataFrame):
            prepped = self.prep(data, output="sparse")
            X = prepped.rhs
            y = prepped.lhs.toarray().ravel()
            self.model.partial_fit(X=X, y=y)
            return self.model
        fitted_model = call_py(stream_data, data=self.data, fn=fitting_fn, last=True, verbose=verbose)
        self.model = fitted_model

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    airline = pl.scan_parquet(base_dir / "data" / "airline")
    llm = LargeLinearRegressor(
        formula="arrival_delay ~ departure_delay + year + day_of_week + scheduled_elapsed_time",
        data=airline,
        query=lambda x: x.filter(pl.col("cancelled").ne(1)),
        batch_size=5000,
        type="ols_robust"
    )
    llm.fit()
    print(f"SGD Coef: {llm.model.coef_}")

    comparison_data = llm.prep(airline.filter(pl.col("cancelled").ne(1)).collect(), output="numpy")
    X = comparison_data.rhs
    y = comparison_data.lhs.ravel()
    results = sm.OLS(y, X).fit()
    print(results.summary())