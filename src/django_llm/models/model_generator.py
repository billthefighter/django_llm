# Configure Django settings before importing any Django-related modules
import os
import django
from django.conf import settings

# Configure Django settings if not already configured
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django_llm",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

# Now import other modules that depend on Django
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator
from llmaestro.core.persistence import PersistentModel
from inspect import isclass

def is_persistent_model(obj):
    """
    Filter function that returns True if a class is a subclass of PersistentModel.
    
    Args:
        obj: The object to check
        
    Returns:
        bool: True if the object is a class and a subclass of PersistentModel, False otherwise
    """
    return isclass(obj) and issubclass(obj, PersistentModel) and obj != PersistentModel


def generate_models():
    """
    Generate Django models from Pydantic models.
    """
    generator = StaticDjangoModelGenerator(
        output_path="src/django_llm/models/models.py",
        packages=["llmaestro"],
        app_label="django_llm",
        filter_function=is_persistent_model,
        )
    generator.generate()

if __name__ == "__main__":
    generate_models()