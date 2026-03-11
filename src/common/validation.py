def require_non_empty_dataframe(input_df, input_name="Input"):
    if input_df is None or input_df.empty:
        raise ValueError(f"{input_name} must contain at least one row.")


def require_columns(df, required_columns, input_name="Input"):
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{input_name} is missing required columns: {missing_columns}")


def get_required_columns_from_preprocessor(preprocessor):
    required_columns = []
    for _, _, column_names in preprocessor.transformers_:
        if isinstance(column_names, (list, tuple)):
            required_columns.extend(column_names)
    return required_columns

