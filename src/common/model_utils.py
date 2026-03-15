import joblib
from functools import lru_cache
from pathlib import Path


def get_repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return get_repo_root() / path


@lru_cache(maxsize=8)
def _load_model_bundle_cached(resolved_model_path):
    path = Path(resolved_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run training first.")

    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model bundle from {path}: {exc}") from exc

    if not isinstance(bundle, (tuple, list)) or len(bundle) != 2:
        raise ValueError(
            f"Model bundle at {path} must contain exactly (model, preprocessor)."
        )

    return bundle[0], bundle[1]


def load_model_bundle(model_path):
    resolved_model_path = resolve_repo_path(model_path)
    return _load_model_bundle_cached(str(resolved_model_path))


def get_positive_class_probability(model, processed_input, positive_class=1, model_name="model"):
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Loaded {model_name} does not support predict_proba().")
    if not hasattr(model, "classes_"):
        raise AttributeError(f"Loaded {model_name} does not expose classes_.")

    class_list = list(model.classes_)
    if positive_class not in class_list:
        raise ValueError(
            f"{model_name} classes do not include positive class {positive_class}: {class_list}"
        )

    if getattr(processed_input, "shape", [0])[0] != 1:
        raise ValueError(
            f"{model_name} scoring expects exactly one row after preprocessing. "
            f"Received shape {getattr(processed_input, 'shape', None)}."
        )

    positive_class_index = class_list.index(positive_class)
    probability = model.predict_proba(processed_input)[0][positive_class_index]
    return float(probability)
