import joblib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def load_model_bundle(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run training first.")
    return joblib.load(path)


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

    positive_class_index = class_list.index(positive_class)
    probability = model.predict_proba(processed_input)[0][positive_class_index]
    return float(probability)

