import formulaic as fml
import math
import polars as pl
from typing import Dict, List


def categorical_enum(
    data: pl.LazyFrame, col: str, ref: str | None = None, cast_to_string: bool = False
) -> pl.Enum:
    """
    Construct an Enum object from unique LazyFrame column values.

    Parameters
    ----------
    df: pl.LazyFrame
        The input LazyFrame.
    col: str
        The column name from which to extract unique values. Should be
        specified as a string.
    ref: str | None
        Optional parameter. The reference level for the unique values specified
        as a string. This is useful for specifying the reference level for
        categorical LazyFrame columns.
    cast_to_string: bool
        Whether to cast the column to a string before exracting unique values.
        This is helpful e.g. when a categorical variable is stored as a numeric
        datatype like an integer.

    Returns
    -------
    pl.Enum
        Returns an Enum of the unique column values. Value `ref` will be the
        first value, if it is specified.
    """
    if cast_to_string:
        query = data.select(pl.col(col).cast(str))
    else:
        query = data.select(pl.col(col))
    vals = (
        query.unique(pl.col(col))
        .sort(pl.col(col))
        .collect(engine="streaming")
        .to_series()
        .to_list()
    )
    if ref:
        if ref not in vals:
            raise ValueError(f'Reference level "{ref}" was not found in column `{col}`')
        vals.remove(ref)
        vals.insert(0, ref)
    return pl.Enum(vals)


def encode_categorical(data: pl.LazyFrame, formula: str) -> pl.LazyFrame:
    schema = split_schema(data)
    string_cols = schema["string"]
    required_variables = formula_terms(formula)
    if required_variables:
        string_cols = [c for c in string_cols if c in required_variables]
    if string_cols:
        for col in string_cols:
            enum = categorical_enum(data=data, col=col)
            data = data.with_columns(pl.col(col).cast(enum).alias(col))
    return data


def formula_terms(formula: str) -> List[str]:
    """Return all non-lhs formula variables"""
    formula = fml.Formula(formula)
    if hasattr(formula, "lhs"):
        lhs_var = formula.lhs.required_variables
        if len(lhs_var) > 1:
            raise ValueError(
                f"Formula lhs must be a single variable; currently {formula.lhs}"
            )
    else:
        lhs_var = []
    try:
        required_vars = list(formula.required_variables)
    except Exception as _:
        required_vars = list()
    required_vars = [x for x in required_vars if x not in lhs_var]
    return required_vars


def lhs(formula: fml.Formula) -> str:
    if not hasattr(formula, "lhs"):
        raise ValueError("The provided formula is missing a response variable")
    y = formula.lhs.required_variables
    if len(y) > 1:
        raise ValueError(
            f"Formula response variable must be a single variable; currently {formula.lhs}"
        )
    return y.pop()


def nrow(df: pl.LazyFrame) -> int:
    return df.select(pl.len()).collect(engine="streaming").item()


def scale_numeric(
    data: pl.DataFrame, stats: Dict[str, Dict[str, float]]
) -> pl.DataFrame:
    if stats:
        for col, stat in stats.items():
            mean = stat["mean"]
            std = stat["std"]
            if math.isclose(std, 0.0):
                std = 1.0
            data = data.with_columns(((pl.col(col) - mean) / std).alias(col))
    return data


def sgd_config_regression(kind: str):
    kind = kind.lower()
    mapping = {
        "ols": ("squared_error", None),
        "ridge": ("squared_error", "l2"),
        "lasso": ("squared_error", "l1"),
        "elasticnet": ("squared_error", "elasticnet"),
        "ols_robust": ("huber", None),
        "huber": ("huber", "l2"),
        "svr_linear": ("epsilon_insensitive", "l2"),
        "svr_squared": ("squared_epsilon_insensitive", "l2"),
    }
    if kind not in mapping:
        raise ValueError(f"Unknown model type: {kind}. Allowed: {list(mapping.keys())}")
    return mapping[kind]


def sgd_config_classification(kind: str):
    kind = kind.lower()
    mapping = {
        # logistic regression
        "logistic": ("log_loss", None),
        "logistic_l1": ("log_loss", "l1"),
        "logistic_l2": ("log_loss", "l2"),
        "logistic_elasticnet": ("log_loss", "elasticnet"),
        # linear SVM
        "svm": ("hinge", "l2"),
        "svm_squared": ("squared_hinge", "l2"),
        "svm_modified": ("modified_huber", "l2"),
        # perceptron variants
        "perceptron": ("perceptron", None),
        # robust losses
        "huber": ("huber", "l2"),
        "huber_l1": ("huber", "l1"),
        "huber_elasticnet": ("huber", "elasticnet"),
    }
    if kind not in mapping:
        raise ValueError(f"Unknown model type: {kind}. Allowed: {list(mapping.keys())}")
    return mapping[kind]


def split_schema(data: pl.LazyFrame | pl.DataFrame) -> Dict[str, List[str]]:
    """
    Given a Polars LazyFrame schema (dict: name -> dtype),
    return (numeric_cols, categorical_cols).

    Rules:
    - categorical = Utf8, Enum, Categorical
    - numeric = Int*, UInt*, Float*
    - anything else = Error
    """
    schema = data.collect_schema()
    numeric = []
    categorical = []
    string = []

    for col, dt in schema.items():
        if dt in (pl.Categorical, pl.Enum):
            categorical.append(col)
        elif isinstance(dt, pl.Utf8):
            string.append(col)
        elif isinstance(
            dt,
            (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            ),
        ):
            numeric.append(col)

    return {"numeric": numeric, "categorical": categorical, "string": string}


def summary_stats(data: pl.LazyFrame, formula: str) -> Dict[str, Dict[str, float]]:
    stats = dict()
    schema = split_schema(data)
    numeric_cols = schema["numeric"]
    required_variables = formula_terms(formula)
    if required_variables:
        numeric_cols = [c for c in numeric_cols if c in required_variables]
    if numeric_cols:
        exprs = []
        for c in numeric_cols:
            exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
            exprs.append(pl.col(c).std().alias(f"{c}_std"))
        out = data.select(exprs).collect(engine="streaming")
        stats = {
            col: {"mean": out[f"{col}_mean"].item(), "std": out[f"{col}_std"].item()}
            for col in numeric_cols
        }
    return stats


def unique(data: pl.LazyFrame, col: str):
    """Get unique values from a LazyFrame column"""
    return (
        data.select(pl.col(col))
        .unique(pl.col(col))
        .sort(pl.col(col))
        .collect(engine="streaming")
        .to_series()
        .to_numpy()
    )
