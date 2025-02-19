"""
Dynamic model discovery and registration for Django-LLM.
"""
from typing import Dict, Type, Optional
import inspect
import logging

from django.contrib import admin
from django.contrib.admin.exceptions import AlreadyRegistered
from django.db import models
from django.apps import apps
from django.core.exceptions import AppRegistryNotReady
from pydantic import BaseModel
from pydantic2django import ModelRegistryManager, make_django_model


# Global registry instance
_registry: Optional[ModelRegistryManager] = None
logger = logging.getLogger(__name__)

# Cache for discovered models
_discovered_pydantic_models: Dict[str, Type[BaseModel]] = {}
_model_dependencies: Dict[str, set[str]] = {}
_registration_order: list[str] = []


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


def analyze_model_dependencies(pydantic_model: Type[BaseModel]) -> set[str]:
    """Analyze dependencies of a Pydantic model without creating Django models."""
    deps = set()
    for field in pydantic_model.model_fields.values():
        annotation = field.annotation
        if annotation is not None:
            # Check for List/Dict types containing Pydantic models
            if hasattr(annotation, "__origin__"):
                origin = getattr(annotation, "__origin__", None)
                if origin in (list, dict):
                    args = getattr(annotation, "__args__", [])
                    for arg in args:
                        if inspect.isclass(arg) and issubclass(arg, BaseModel):
                            deps.add(arg.__name__)
            # Check for direct Pydantic model references
            elif inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                deps.add(annotation.__name__)
    return deps


def discover_models(app_label: str = "django_llm") -> None:
    """Discover and analyze Pydantic models without creating Django models yet."""
    global _discovered_pydantic_models, _model_dependencies, _registration_order
    
    if _discovered_pydantic_models:
        return  # Already discovered
        
    logger.info(f"Discovering models for {app_label}...")
    
    # Get registry and discover models
    registry = get_registry()
    pydantic_models = list(registry.discover_models("llmaestro", app_label=app_label).values())
    additional_models = list(registry.discover_models("django_llm", app_label=app_label).values())
    
    all_models = pydantic_models + additional_models
    _discovered_pydantic_models = {model.__name__: model for model in all_models}
    
    # Build dependency graph
    _model_dependencies = {}
    for model in all_models:
        _model_dependencies[model.__name__] = analyze_model_dependencies(model)
    
    # Determine registration order
    _registration_order = []
    registered = set()
    
    def register_with_deps(model_name: str, visited: set[str] | None = None):
        if visited is None:
            visited = set()
        if model_name in visited:
            logger.warning(f"Circular dependency detected for {model_name}")
            return
        if model_name in registered:
            return
        
        visited.add(model_name)
        for dep in _model_dependencies[model_name]:
            if dep in _discovered_pydantic_models:
                register_with_deps(dep, visited)
        visited.remove(model_name)
        
        if model_name not in registered:
            _registration_order.append(model_name)
            registered.add(model_name)
    
    for model_name in _discovered_pydantic_models:
        register_with_deps(model_name)
    
    logger.info(f"Discovered {len(_discovered_pydantic_models)} models with registration order determined")


def setup_dynamic_models(app_label: str = "django_llm", skip_admin: bool = False) -> Dict[str, Type[models.Model]]:
    """Set up dynamic models for Django-LLM.
    
    This function should be called during migration operations, not during app initialization.
    It creates Django models from discovered Pydantic models.
    """
    # Ensure models are discovered
    if not _discovered_pydantic_models:
        discover_models(app_label)
        
    django_models = {}
    registered_models = set()
    
    try:
        # Get currently registered models if possible
        registered_models = set(
            f"{model._meta.app_label}.{model._meta.model_name}"
            for model in apps.get_app_config(app_label).get_models()
        )
    except Exception:
        logger.debug("Could not get registered models, will attempt to create all")
    
    # First pass: Create all models without relationships
    logger.info("First pass: Creating models without relationships...")
    for model_name in _registration_order:
        model = _discovered_pydantic_models[model_name]
        try:
            if model_name.startswith("Django"):
                model_name = model_name[6:]

            # Check if model is already registered
            model_key = f"{app_label}.{model_name.lower()}"
            if model_key in registered_models:
                try:
                    existing = apps.get_model(app_label, model_name)
                    logger.info(f"Model {model_name} already registered, skipping")
                    django_models[model_name] = existing
                    continue
                except Exception:
                    logger.debug(f"Could not get existing model {model_name}, will recreate")

            # Log model fields before creation
            logger.debug(f"Creating model {model_name} with fields:")
            for field_name, field in model.model_fields.items():
                logger.debug(f"  - {field_name}: {field.annotation}")

            try:
                # Create Django model without relationships first
                logger.debug(f"Attempting to create Django model for {model_name}")
                result = make_django_model(
                    model,
                    app_label=app_label,
                    db_table=f"{app_label}_{model_name.lower()}",
                    skip_relationships=True,  # Skip relationships in first pass
                )
                django_model = result[0]
                logger.debug(f"Successfully created Django model class for {model_name}")
            except Exception as e:
                logger.error(f"Failed to create Django model class for {model_name}: {str(e)}")
                logger.error(f"Model definition: {model}")
                raise

            try:
                # Register model with Django
                logger.debug(f"Attempting to register {model_name} with Django")
                apps.register_model(app_label, django_model)
                django_models[model_name] = django_model
                logger.info(f"Created and registered base model for {model.__name__}")
            except Exception as e:
                logger.error(f"Failed to register {model_name} with Django: {str(e)}")
                logger.error(f"Model class: {django_model}")
                raise

        except Exception as e:
            logger.error(f"Error creating base model for {model.__name__}: {str(e)}")
            logger.error("Full model creation traceback:", exc_info=True)
            raise  # Re-raise to prevent partial model registration

    # Second pass: Add relationships now that all models exist
    logger.info("Second pass: Adding relationships between models...")
    for model_name in _registration_order:
        if model_name.startswith("Django"):
            model_name = model_name[6:]
            
        try:
            model = _discovered_pydantic_models[model_name]
            django_model = django_models[model_name]
            
            # Log relationship info
            logger.debug(f"Processing relationships for {model_name}")
            deps = analyze_model_dependencies(model)
            if deps:
                logger.debug(f"Model {model_name} has dependencies: {deps}")
            
            # Add relationships
            try:
                logger.debug(f"Creating relationship fields for {model_name}")
                result = make_django_model(
                    model,
                    app_label=app_label,
                    db_table=f"{app_label}_{model_name.lower()}",
                    skip_relationships=False,
                    existing_model=django_model
                )
                
                # make_django_model returns (model, fields) tuple
                field_updates = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                
                if field_updates:
                    if isinstance(field_updates, dict):
                        logger.debug(f"Got field updates for {model_name}: {list(field_updates.keys())}")
                    else:
                        logger.debug(f"Got field updates for {model_name} of unexpected type: {type(field_updates)}")
                
                # Apply relationship fields if any were returned
                if isinstance(field_updates, dict):
                    for field_name, field in field_updates.items():
                        if not hasattr(django_model, field_name):
                            logger.debug(f"Adding field {field_name} to {model_name}")
                            field.contribute_to_class(django_model, field_name)
                            logger.info(f"Added relationship field {field_name} to {model_name}")
                        else:
                            logger.debug(f"Field {field_name} already exists on {model_name}")
            except Exception as e:
                logger.error(f"Failed to create relationship fields for {model_name}: {str(e)}")
                logger.error("Full relationship creation traceback:", exc_info=True)
                raise
                    
        except Exception as e:
            logger.error(f"Error adding relationships to {model_name}: {str(e)}")
            logger.error("Full relationship traceback:", exc_info=True)
            raise  # Re-raise to prevent partial relationship setup

    # Register with admin if requested
    if not skip_admin:
        for model_name, django_model in django_models.items():
            try:
                admin.site.register(django_model, DynamicModelAdmin)
                logger.info(f"Registered {model_name} with admin")
            except AlreadyRegistered:
                logger.info(f"Admin interface for {model_name} already registered")

    logger.info(f"Successfully registered {len(django_models)} models with Django")
    return django_models


def register_model_admins(app_label: str = "django_llm") -> None:
    """Register admin interfaces for all models in the app.
    
    This should be called after Django is fully initialized and migrations are complete.
    """
    logger.info(f"Registering admin interfaces for {app_label}...")
    
    for model in apps.get_app_config(app_label).get_models():
        try:
            admin.site.register(model, DynamicModelAdmin)
            logger.info(f"Registered admin interface for {model.__name__}")
        except AlreadyRegistered:
            logger.info(f"Admin interface for {model.__name__} already registered")


def get_discovered_models() -> Dict[str, Type[BaseModel]]:
    """Get all discovered Pydantic models."""
    return _discovered_pydantic_models


def get_django_models() -> Dict[str, Type[models.Model]]:
    """Get all registered Django models."""
    return {name: apps.get_model("django_llm", name) 
            for name in _discovered_pydantic_models.keys()
            if not name.startswith("Django")} 