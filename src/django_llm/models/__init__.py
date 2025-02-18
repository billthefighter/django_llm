"""
Django-LLM models package with dynamic model discovery.
"""
from typing import Dict, Type

from django.db import models
from pydantic import BaseModel

from .model_discovery import (
    setup_dynamic_models,
    get_discovered_models,
    get_django_models,
    get_registry,
)

# Initialize registry and discover models early
registry = get_registry()

# Dictionary to hold all our models
discovered_models: Dict[str, Type[BaseModel]] = {}
django_models: Dict[str, Type[models.Model]] = {}

def _update_model_refs():
    """Update module level model references."""
    global discovered_models, django_models
    discovered_models.update(get_discovered_models())
    django_models.update(get_django_models())

# Initial model discovery and registration - do this as early as possible
django_models = setup_dynamic_models("django_llm")
_update_model_refs()

# Make all models available in the module namespace
globals().update(django_models)

# Export the models and utility functions
__all__ = [
    "setup_dynamic_models",
    "get_discovered_models",
    "get_django_models",
    *list(django_models.keys())  # Add all model names
]

# For debugging - print discovered models
print(f"Django-LLM: Discovered and registered {len(django_models)} models:")
for model_name, model_class in django_models.items():
    print(f"  - {model_name}: {model_class._meta.db_table}") 