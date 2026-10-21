
from typing import Any, Union, Type

def model_name(model: Any) -> str:
    """Extracts name attribute or class name from a model."""
    return getattr(model, "name", model.__class__.__name__)

def attack_name(attack: Union[Type, Any]) -> str:
    """Extracts name from an attack class or instance."""
    if isinstance(attack, type):
        return attack.__name__
    return getattr(attack, "attack_name", attack.__class__.__name__)