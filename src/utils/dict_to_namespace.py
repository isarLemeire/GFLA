from types import SimpleNamespace
from typing import Any, Dict

def dict_to_namespace(data: Dict[str, Any]) -> SimpleNamespace:
    """Recursively converts a dictionary to a SimpleNamespace object,

    scaling 'eps' and 'alpha' (or lists of them) by 255 if passed as raw magnitudes >= 1.0.
    """
    scale_keys = {"eps", "alpha"}
    processed_dict = {}

    for k, v in data.items():
        # Scale targeted keys if numeric
        if k in scale_keys:
            if isinstance(v, (int, float)) and v >= 1.0:
                v = v / 255.0
            elif isinstance(v, list):
                v = [
                    item / 255.0 if isinstance(item, (int, float)) and item >= 1.0 else item
                    for item in v
                ]

        # Recursive steps
        if isinstance(v, dict):
            processed_dict[k] = dict_to_namespace(v)
        elif isinstance(v, list):
            processed_dict[k] = [
                dict_to_namespace(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            processed_dict[k] = v

    return SimpleNamespace(**processed_dict)