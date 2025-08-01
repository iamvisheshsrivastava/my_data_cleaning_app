from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

@dataclass
class SMOTEParams:
    target_column: str
    sampling_strategy: str = "auto"
    k_neighbors: int = 5

@dataclass
class OversampleParams:
    target_column: str
    sampling_strategy: str = "minority"

@dataclass
class UndersampleParams:
    target_column: str
    sampling_strategy: str = "majority"

@dataclass
class ImputeMissingValuesParams:
    strategy: str = "mean"
    fill_value: float = 0.0

@dataclass
class NormalizeParams:
    pass

@dataclass
class ScaleParams:
    min_value: int = 0
    max_value: int = 1

@dataclass
class LLMCleaningParams:
    code: str

@dataclass
class BinarizeParams:
    threshold: float = 0.0

@dataclass
class RemoveOutliersParams:
    iqr_multiplier: float = 1.5

@dataclass
class RemoveNaNColsParams:
    nan_threshold: float = 0.5

@dataclass
class RemoveNaNRowsParams:
    nan_threshold: float = 0.5

@dataclass
class DatasetMultiplierParams:
    size_multiplier: int = 1

@dataclass
class AddNoiseParams:
    noise_factor: float = 0.1

def impute_missing(data: pd.DataFrame, params: ImputeMissingValuesParams) -> pd.DataFrame:
    strategy = params.strategy
    fill_value = params.fill_value
    numeric_cols = data.select_dtypes(include=np.number).columns
    if strategy in ["mean", "median", "constant"]:
        if numeric_cols.empty:
            raise ValueError(f"No numeric columns found for strategy '{strategy}'.")
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    elif strategy == "most_frequent":
        imputer = SimpleImputer(strategy=strategy)
        data[data.columns] = imputer.fit_transform(data)
    else:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")
    return data


def normalize(data: pd.DataFrame, params: NormalizeParams) -> pd.DataFrame:
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

def scale(data: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=(params.min_value, params.max_value))
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

def llm_data_cleaning(data: pd.DataFrame, params: Any) -> pd.DataFrame:
    """
    Applies LLM-generated dynamic cleaning code to the given DataFrame.
    Expects 'code' inside params, which should be a string of Python code
    that modifies the variable `df` in-place.
    """
    code = getattr(params, "code", None)
    if not code:
        raise ValueError("LLM Data Cleaning failed: No code provided in step parameters.")

    try:
        local_vars = {"df": data.copy()}
        global_vars = {
            "pd": pd,
            "np": np,
            "SimpleImputer": SimpleImputer,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "Binarizer": Binarizer,
            "SMOTE": SMOTE,
            "RandomOverSampler": RandomOverSampler,
            "RandomUnderSampler": RandomUnderSampler
        }
        exec(code, global_vars, local_vars)
        return local_vars["df"]
    except Exception as e:
        raise RuntimeError(
            f"LLM Data Cleaning failed to execute.\n\n"
            f"Please verify the generated code or revise the cleaning suggestion.\n\n"
            f"Error: {str(e)}"
        )

def one_hot_encode(data: pd.DataFrame, params: Any) -> pd.DataFrame:
    categorical_cols = data.select_dtypes(include='object').columns
    return pd.get_dummies(data, columns=categorical_cols)

def binarize(data: pd.DataFrame, params: BinarizeParams) -> pd.DataFrame:
    binarizer = Binarizer(threshold=params.threshold)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if data[numeric_cols].isnull().any().any():
        raise ValueError(
            f"Binarization failed: Numeric columns contain NaNs. "
            f"Please impute missing values first. Columns: "
            f"{data[numeric_cols].columns[data[numeric_cols].isnull().any()].tolist()}"
        )
    data[numeric_cols] = binarizer.fit_transform(data[numeric_cols])
    return data

def remove_outliers(data: pd.DataFrame, params: RemoveOutliersParams) -> pd.DataFrame:
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - params.iqr_multiplier * IQR
        upper = Q3 + params.iqr_multiplier * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

def remove_nan_cols(data: pd.DataFrame, params: RemoveNaNColsParams) -> pd.DataFrame:
    return data.loc[:, data.isnull().mean() < params.nan_threshold]

def remove_nan_rows(data: pd.DataFrame, params: RemoveNaNRowsParams) -> pd.DataFrame:
    return data.loc[data.isnull().mean(axis=1) < params.nan_threshold]

def dataset_multiplier(data: pd.DataFrame, params: DatasetMultiplierParams) -> pd.DataFrame:
    try:
        multiplier = int(str(params.size_multiplier).replace("x", ""))
    except Exception:
        raise ValueError(
            f"Invalid multiplier value: '{params.size_multiplier}'. "
            "Expected format like '1x', '2x', ..., '5x'."
        )
    return pd.concat([data] * multiplier, ignore_index=True)

def add_noise(data: pd.DataFrame, params: AddNoiseParams) -> pd.DataFrame:
    numeric_cols = data.select_dtypes(include=np.number).columns
    noise = np.random.normal(loc=0.0, scale=params.noise_factor, size=data[numeric_cols].shape)
    data[numeric_cols] = data[numeric_cols] + noise
    return data

def apply_smote(data: pd.DataFrame, params: SMOTEParams) -> pd.DataFrame:
    if params.target_column not in data.columns:
        raise ValueError(f"SMOTE: Target column '{params.target_column}' not found.")
    X = data.drop(columns=[params.target_column])
    y = data[params.target_column]
    smote = SMOTE(sampling_strategy=params.sampling_strategy, k_neighbors=params.k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=params.target_column)], axis=1)

def oversample_data(data: pd.DataFrame, params: OversampleParams) -> pd.DataFrame:
    if params.target_column not in data.columns:
        raise ValueError(f"Oversampling: Target column '{params.target_column}' not found.")
    y = data[params.target_column]
    X = data.drop(columns=[params.target_column])
    if y.isnull().any():
        raise ValueError(
            f"Oversampling skipped: Target column '{params.target_column}' contains missing values (NaNs). "
            "Please impute or remove them before applying oversampling."
        )
    try:
        y = y.astype(str)
        sampler = RandomOverSampler(sampling_strategy=params.sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=params.target_column)], axis=1)
    except Exception as e:
        raise ValueError(
            f"Oversampling failed. Ensure the target column contains consistent class labels (no mixed types).\n\n"
            f"Details: {str(e)}"
        )

def undersample_data(data: pd.DataFrame, params: UndersampleParams) -> pd.DataFrame:
    if params.target_column not in data.columns:
        raise ValueError(f"Undersampling: Target column '{params.target_column}' not found.")
    y = data[params.target_column]
    X = data.drop(columns=[params.target_column])
    if y.isnull().any():
        raise ValueError(
            f"Undersampling skipped: Target column '{params.target_column}' contains missing values (NaNs). "
            "Please clean or impute the target column before applying undersampling."
        )
    try:
        y = y.astype(str)
        sampler = RandomUnderSampler(sampling_strategy=params.sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.concat(
            [pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=params.target_column)],
            axis=1
        )
    except Exception as e:
        raise ValueError(
            f"Undersampling failed. Make sure the target column contains clean and consistent class labels.\n\n"
            f"Details: {str(e)}"
        )

OPERATION_MAP = {
    "Impute Missing Values": impute_missing,
    "Normalize": normalize,
    "Scale": scale,
    "LLM Data Cleaning": llm_data_cleaning,
    "One-Hot Encoding": one_hot_encode,
    "Binarize": binarize,
    "Remove Outliers": remove_outliers,
    "Remove Columns with Excessive NaNs": remove_nan_cols,
    "Remove Rows with Excessive NaNs": remove_nan_rows,
    "SMOTE": apply_smote,
    "Dataset Multiplier": dataset_multiplier,
    "Add Noise": add_noise,
    "Oversample": oversample_data,
    "Undersample": undersample_data
}

def run_pipeline(data: pd.DataFrame, steps: List[Dict[str, Any]]) -> pd.DataFrame:
    for step in steps:
        name = step["name"]
        params_dict = step.get("params", {})

        if name == "Impute Missing Values":
            params = ImputeMissingValuesParams(**params_dict)
        elif name == "Normalize":
            params = NormalizeParams()
        elif name == "Scale":
            params = ScaleParams(**params_dict)
        elif name == "LLM Data Cleaning":
            params = LLMCleaningParams(**params_dict)
        elif name == "One-Hot Encoding":
            params = {}
        elif name == "Binarize":
            params = BinarizeParams(**params_dict)
        elif name == "Remove Outliers":
            params = RemoveOutliersParams(**params_dict)
        elif name == "Remove Columns with Excessive NaNs":
            params = RemoveNaNColsParams(**params_dict)
        elif name == "Remove Rows with Excessive NaNs":
            params = RemoveNaNRowsParams(**params_dict)
        elif name == "SMOTE":
            params = SMOTEParams(**params_dict)
        elif name == "Dataset Multiplier":
            params = DatasetMultiplierParams(**params_dict)
        elif name == "Add Noise":
            params = AddNoiseParams(**params_dict)
        elif name == "Oversample":
            params = OversampleParams(**params_dict)
        elif name == "Undersample":
            params = UndersampleParams(**params_dict)
        else:
            continue  
        data = OPERATION_MAP[name](data, params)

    return data
