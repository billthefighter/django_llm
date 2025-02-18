"""
Dynamic model discovery and registration for Django-LLM.
"""
from typing import Dict, Type, Optional

from django.contrib import admin
from django.contrib.admin.exceptions import AlreadyRegistered
from django.db import models
from django.apps import apps
from pydantic import BaseModel
from pydantic2django import ModelRegistryManager


# Global registry instance
_registry: Optional[ModelRegistryManager] = None


def get_registry() -> ModelRegistryManager:
    """Get or create the global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistryManager()
    return _registry


class DynamicModelAdmin(admin.ModelAdmin):
    """Dynamic admin interface for discovered models."""
    list_display = ('id',)  # Start with basic fields
    search_fields = ('id',)
    
    def get_list_display(self, request):
        """Dynamically determine which fields to display."""
        model = self.model
        # Start with id field
        fields = ['id']
        # Add other fields that might be interesting
        for field in model._meta.fields:
            if field.name != 'id':
                fields.append(field.name)
        return fields


def setup_dynamic_models(app_label: str = "django_llm") -> Dict[str, Type[models.Model]]:
    """
    Setup dynamic models for the app, including discovery, registration and admin setup.
    
    Args:
        app_label: The Django app label to use
        
    Returns:
        Dict mapping model names to Django model classes
    """
    # Get or create registry
    registry = get_registry()
    
    # Discover models
    discovered = registry.discover_models(app_label)
    
    # Register models
    django_models = registry.register_models(app_label=app_label)
    
    # Ensure models are properly registered in Django's app registry
    for model_name, model_class in django_models.items():
        try:
            # Get model name without Django prefix
            model_name_str = model_class._meta.model_name
            if not model_name_str:
                continue
                
            # Ensure model is properly configured
            if not hasattr(model_class._meta, 'app_label'):
                model_class._meta.app_label = app_label
                
            if not hasattr(model_class._meta, 'db_table'):
                model_class._meta.db_table = f"{app_label}_{model_name_str}"
                
            # Check if model is already registered
            try:
                existing = apps.get_registered_model(app_label, model_name_str)
                if existing is not model_class:
                    # Re-register if it's a different instance
                    apps.all_models[app_label].pop(model_name_str, None)
                    apps.register_model(app_label, model_class)
            except LookupError:
                # Model not registered yet
                apps.register_model(app_label, model_class)
                
            # Register with admin
            try:
                admin.site.register(model_class, DynamicModelAdmin)
            except AlreadyRegistered:
                pass
                
        except Exception as e:
            print(f"Warning: Could not register model {model_name}: {e}")
            continue
            
    return django_models


def get_discovered_models() -> Dict[str, Type[BaseModel]]:
    """Get all discovered Pydantic models."""
    registry = get_registry()
    return registry.discovered_models


def get_django_models() -> Dict[str, Type[models.Model]]:
    """Get all registered Django models."""
    registry = get_registry()
    return registry.django_models 