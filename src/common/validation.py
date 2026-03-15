import logging

import pandas as pd
from src.common.logging_utils import get_logger


logger = get_logger(__name__)


def ensure_dataframe(input_data, input_name="Input"):
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    if isinstance(input_data, pd.Series):
        return input_data.to_frame().T
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    raise TypeError(f"{input_name} must be a pandas DataFrame, Series, or dict.")


def require_non_empty_dataframe(input_df, input_name="Input"):
    if input_df is None or input_df.empty:
        raise ValueError(f"{input_name} must contain at least one row.")


def require_single_row_dataframe(input_df, input_name="Input"):
    if len(input_df) != 1:
        raise ValueError(
            f"{input_name} must contain exactly one row for single-record scoring. "
            f"Received {len(input_df)} rows."
        )


def require_columns(df, required_columns, input_name="Input"):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        logger.warning("%s is missing required columns: %s", input_name, missing_columns)
        raise ValueError(f"{input_name} is missing required columns: {missing_columns}")


def align_features(input_df, required_columns, input_name="Input"):
    require_columns(input_df, required_columns, input_name=input_name)

    extra_columns = [column for column in input_df.columns if column not in required_columns]
    if extra_columns:
        logger.warning(
            "%s contains extra columns that will be ignored: %s",
            input_name,
            extra_columns,
        )

    return input_df.loc[:, required_columns].copy()


def get_required_columns_from_preprocessor(preprocessor):
    required_columns = []
    for _, _, column_names in preprocessor.transformers_:
        if isinstance(column_names, (list, tuple)):
            required_columns.extend(column_names)
    return required_columns


def get_preprocessor_feature_metadata(preprocessor):
    metadata = {
        "required_columns": [],
        "numerical_columns": [],
        "categorical_columns": [],
        "categorical_allowed_values": {},
    }

    for name, transformer, column_names in getattr(preprocessor, "transformers_", []):
        if not isinstance(column_names, (list, tuple)):
            continue

        column_list = list(column_names)
        metadata["required_columns"].extend(column_list)

        if name == "num":
            metadata["numerical_columns"].extend(column_list)
        elif name == "cat":
            metadata["categorical_columns"].extend(column_list)
            encoder = getattr(transformer, "named_steps", {}).get("encoder")
            if encoder is not None and hasattr(encoder, "categories_"):
                for column_name, categories in zip(column_list, encoder.categories_):
                    metadata["categorical_allowed_values"][column_name] = {
                        str(category) for category in categories.tolist()
                    }

    return metadata


def _is_missing_value(value):
    if pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def validate_record_schema(
    row_payload,
    required_columns,
    numerical_columns=None,
    categorical_allowed_values=None,
    input_name="Input",
):
    numerical_columns = numerical_columns or []
    categorical_allowed_values = categorical_allowed_values or {}
    validation_errors = []

    missing_columns = [
        column_name for column_name in required_columns if column_name not in row_payload
    ]
    if missing_columns:
        validation_errors.append(
            f"{input_name} is missing required columns: {missing_columns}"
        )
        return validation_errors

    for column_name in required_columns:
        value = row_payload.get(column_name)
        if _is_missing_value(value):
            validation_errors.append(
                f"{input_name} has null or empty value for required column '{column_name}'"
            )

    for column_name in numerical_columns:
        value = row_payload.get(column_name)
        if _is_missing_value(value):
            continue

        converted_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(converted_value):
            validation_errors.append(
                f"{input_name} has non-numeric value for '{column_name}': {value}"
            )

    for column_name, allowed_values in categorical_allowed_values.items():
        value = row_payload.get(column_name)
        if _is_missing_value(value):
            continue

        if str(value) not in allowed_values:
            validation_errors.append(
                f"{input_name} has invalid category for '{column_name}': {value}"
            )

    return validation_errors


def warn_on_unseen_categories(input_df, preprocessor, input_name="Input"):
    categorical_transformer = None
    categorical_columns = None

    for name, transformer, column_names in getattr(preprocessor, "transformers_", []):
        if name == "cat" and isinstance(column_names, (list, tuple)):
            categorical_transformer = transformer
            categorical_columns = list(column_names)
            break

    if not categorical_transformer or not categorical_columns:
        return

    encoder = getattr(categorical_transformer, "named_steps", {}).get("encoder")
    if encoder is None or not hasattr(encoder, "categories_"):
        return

    for column_name, known_categories in zip(categorical_columns, encoder.categories_):
        observed_values = {
            str(value)
            for value in input_df[column_name].dropna().unique().tolist()
        }
        trained_values = {str(value) for value in known_categories.tolist()}
        unseen_values = sorted(observed_values - trained_values)

        if unseen_values:
            logger.warning(
                "%s contains unseen categories for '%s': %s. They will be ignored by the encoder.",
                input_name,
                column_name,
                unseen_values,
            )
