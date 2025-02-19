"""
Django-LLM models package with dynamic model discovery.
"""
from typing import Dict, Type

from django.db import models
from pydantic import BaseModel
from django.apps import apps

# Import llmaestro to ensure its models are available
import llmaestro  # noqa

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
    discovered_models.clear()
    django_models.clear()
    discovered_models.update(get_discovered_models())
    django_models.update(get_django_models())

# Initialize dynamic models
django_models: Dict[str, Type[models.Model]] = {}
try:
    django_models = setup_dynamic_models() or {}
except Exception as e:
    print(f"Error setting up dynamic models: {e}")

# Register models with Django
if django_models:
    for model_name, model_class in django_models.items():
        try:
            apps.register_model('django_llm', model_class)
        except Exception as e:
            print(f"Error registering model {model_name}: {e}")

# Make all models available in the module namespace
for model_name, model_class in django_models.items():
    globals()[model_name] = model_class

# Export everything
__all__ = [
    "setup_dynamic_models",
    "get_discovered_models",
    "get_django_models",
    "discovered_models",
    "django_models",
    *list(django_models.keys())
]

# For debugging - print discovered models
print(f"Django-LLM: Discovered and registered {len(django_models)} models:")
for model_name, model_class in django_models.items():
    print(f"  - {model_name}: {model_class._meta.db_table}") 