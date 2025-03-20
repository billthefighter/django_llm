"""Django tools for LLMaestro.

This module provides tools for interacting with Django models and applications.
It includes tools for:
- Listing available Django apps/modules
- Listing models in a module
- Getting information about specific model instances
- Creating model instances
- Updating model instances
- Searching for similar model instances
"""

from typing import Any, Dict, List, Optional, Type, Union, Callable, cast
import inspect
import json
from pydantic import BaseModel, Field, validator, create_model
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.core.exceptions import ObjectDoesNotExist, ValidationError, FieldDoesNotExist
from django.forms.models import model_to_dict

from llmaestro.tools.core import BasicFunctionGuard, ToolParams


class DjangoAppListParams(BaseModel):
    """Parameters for listing Django apps."""
    
    include_models: bool = Field(
        default=False,
        description="Whether to include models for each app in the response."
    )


class DjangoModelListParams(BaseModel):
    """Parameters for listing models in a Django app."""
    
    app_label: str = Field(
        description="The label of the Django app to list models from."
    )
    
    @validator('app_label')
    def validate_app_label(cls, v):
        """Validate that the app label exists."""
        if not v:
            raise ValueError("App label cannot be empty")
        if v not in apps.app_configs:
            raise ValueError(f"App '{v}' does not exist")
        return v


class DjangoModelInfoParams(BaseModel):
    """Parameters for getting information about a Django model."""
    
    app_label: str = Field(
        description="The label of the Django app containing the model."
    )
    model_name: str = Field(
        description="The name of the model to get information about."
    )
    pk: Optional[Any] = Field(
        default=None,
        description="The primary key of the model instance to get information about. If not provided, returns model schema."
    )
    
    @validator('app_label')
    def validate_app_label(cls, v):
        """Validate that the app label exists."""
        if not v:
            raise ValueError("App label cannot be empty")
        if v not in apps.app_configs:
            raise ValueError(f"App '{v}' does not exist")
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v, values):
        """Validate that the model name exists in the app."""
        if not v:
            raise ValueError("Model name cannot be empty")
        
        app_label = values.get('app_label')
        if app_label and app_label in apps.app_configs:
            try:
                apps.get_model(app_label, v)
            except LookupError:
                raise ValueError(f"Model '{v}' does not exist in app '{app_label}'")
        return v


class DjangoModelCreateParams(BaseModel):
    """Parameters for creating a Django model instance."""
    
    app_label: str = Field(
        description="The label of the Django app containing the model."
    )
    model_name: str = Field(
        description="The name of the model to create an instance of."
    )
    data: Dict[str, Any] = Field(
        description="The data to use for creating the model instance."
    )
    
    @validator('app_label')
    def validate_app_label(cls, v):
        """Validate that the app label exists."""
        if not v:
            raise ValueError("App label cannot be empty")
        if v not in apps.app_configs:
            raise ValueError(f"App '{v}' does not exist")
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v, values):
        """Validate that the model name exists in the app."""
        if not v:
            raise ValueError("Model name cannot be empty")
        
        app_label = values.get('app_label')
        if app_label and app_label in apps.app_configs:
            try:
                apps.get_model(app_label, v)
            except LookupError:
                raise ValueError(f"Model '{v}' does not exist in app '{app_label}'")
        return v


class DjangoModelUpdateParams(BaseModel):
    """Parameters for updating a Django model instance."""
    
    app_label: str = Field(
        description="The label of the Django app containing the model."
    )
    model_name: str = Field(
        description="The name of the model to update an instance of."
    )
    pk: Any = Field(
        description="The primary key of the model instance to update."
    )
    data: Dict[str, Any] = Field(
        description="The data to use for updating the model instance."
    )
    
    @validator('app_label')
    def validate_app_label(cls, v):
        """Validate that the app label exists."""
        if not v:
            raise ValueError("App label cannot be empty")
        if v not in apps.app_configs:
            raise ValueError(f"App '{v}' does not exist")
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v, values):
        """Validate that the model name exists in the app."""
        if not v:
            raise ValueError("Model name cannot be empty")
        
        app_label = values.get('app_label')
        if app_label and app_label in apps.app_configs:
            try:
                apps.get_model(app_label, v)
            except LookupError:
                raise ValueError(f"Model '{v}' does not exist in app '{app_label}'")
        return v


class DjangoModelSearchParams(BaseModel):
    """Parameters for searching for Django model instances."""
    
    app_label: str = Field(
        description="The label of the Django app containing the model."
    )
    model_name: str = Field(
        description="The name of the model to search for instances of."
    )
    filters: Dict[str, Any] = Field(
        description="The filters to use for searching for model instances. Keys are field names, values are field values."
    )
    limit: int = Field(
        default=10,
        description="The maximum number of results to return."
    )
    
    @validator('app_label')
    def validate_app_label(cls, v):
        """Validate that the app label exists."""
        if not v:
            raise ValueError("App label cannot be empty")
        if v not in apps.app_configs:
            raise ValueError(f"App '{v}' does not exist")
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v, values):
        """Validate that the model name exists in the app."""
        if not v:
            raise ValueError("Model name cannot be empty")
        
        app_label = values.get('app_label')
        if app_label and app_label in apps.app_configs:
            try:
                apps.get_model(app_label, v)
            except LookupError:
                raise ValueError(f"Model '{v}' does not exist in app '{app_label}'")
        return v
    
    @validator('limit')
    def validate_limit(cls, v):
        """Validate that the limit is positive."""
        if v <= 0:
            raise ValueError("Limit must be positive")
        return v


class DjangoGuard(BasicFunctionGuard):
    """Base function guard for Django operations.
    
    This guard provides common functionality for Django operations.
    """
    
    def __init__(self, func: Callable):
        """Initialize the Django guard.
        
        Args:
            func: The function to guard
        """
        super().__init__(func)
    
    def _get_model_class(self, app_label: str, model_name: str) -> Type[models.Model]:
        """Get a Django model class.
        
        Args:
            app_label: The label of the Django app containing the model
            model_name: The name of the model
            
        Returns:
            The Django model class
            
        Raises:
            ValueError: If the model does not exist
        """
        try:
            return apps.get_model(app_label, model_name)
        except LookupError:
            raise ValueError(f"Model '{model_name}' does not exist in app '{app_label}'")
    
    def _get_model_instance(self, model_class: Type[models.Model], pk: Any) -> models.Model:
        """Get a Django model instance.
        
        Args:
            model_class: The Django model class
            pk: The primary key of the model instance
            
        Returns:
            The Django model instance
            
        Raises:
            ValueError: If the model instance does not exist
        """
        try:
            return model_class.objects.get(pk=pk)
        except ObjectDoesNotExist:
            raise ValueError(f"Instance with primary key '{pk}' does not exist for model '{model_class.__name__}'")
    
    def _model_to_dict(self, instance: models.Model) -> Dict[str, Any]:
        """Convert a Django model instance to a dictionary.
        
        Args:
            instance: The Django model instance
            
        Returns:
            A dictionary representation of the model instance
        """
        data = model_to_dict(instance)
        
        # Convert non-serializable values to strings
        for key, value in data.items():
            if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
                data[key] = str(value)
        
        return data
    
    def _get_model_fields(self, model_class: Type[models.Model]) -> Dict[str, Dict[str, Any]]:
        """Get information about the fields of a Django model.
        
        Args:
            model_class: The Django model class
            
        Returns:
            A dictionary mapping field names to field information
        """
        fields = {}
        
        for field in model_class._meta.get_fields():
            # Skip reverse relations
            if field.is_relation and field.auto_created and not field.concrete:
                continue
                
            field_info = {
                "type": field.__class__.__name__,
                "required": not field.blank and not field.null,
            }
            
            # Add field-specific information
            if hasattr(field, "max_length") and field.max_length:
                field_info["max_length"] = field.max_length
                
            if hasattr(field, "choices") and field.choices:
                field_info["choices"] = dict(field.choices)
                
            if field.is_relation:
                field_info["relation"] = {
                    "model": field.related_model.__name__,
                    "app_label": field.related_model._meta.app_label,
                }
                
            fields[field.name] = field_info
            
        return fields


class DjangoAppListGuard(DjangoGuard):
    """Function guard for listing Django apps."""
    
    def __init__(self):
        """Initialize the Django app list guard."""
        super().__init__(self._list_apps)
    
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if it's safe to list Django apps.
        
        Args:
            **kwargs: The arguments to the function
            
        Returns:
            True (listing apps is always safe)
        """
        return True
    
    def _list_apps(self, include_models: bool = False) -> List[Dict[str, Any]]:
        """List Django apps.
        
        Args:
            include_models: Whether to include models for each app
            
        Returns:
            A list of dictionaries containing information about Django apps
        """
        result = []
        
        for app_label, app_config in apps.app_configs.items():
            app_info = {
                "app_label": app_label,
                "name": app_config.name,
                "verbose_name": str(app_config.verbose_name),
            }
            
            if include_models:
                app_info["models"] = []
                
                for model in app_config.get_models():
                    model_info = {
                        "name": model.__name__,
                        "verbose_name": str(model._meta.verbose_name),
                        "verbose_name_plural": str(model._meta.verbose_name_plural),
                    }
                    
                    app_info["models"].append(model_info)
                    
            result.append(app_info)
            
        return result


class DjangoModelListGuard(DjangoGuard):
    """Function guard for listing Django models in an app."""
    
    def __init__(self):
        """Initialize the Django model list guard."""
        super().__init__(self._list_models)
    
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if it's safe to list Django models.
        
        Args:
            **kwargs: The arguments to the function
            
        Returns:
            True (listing models is always safe)
        """
        return True
    
    def _list_models(self, app_label: str) -> List[Dict[str, Any]]:
        """List Django models in an app.
        
        Args:
            app_label: The label of the Django app to list models from
            
        Returns:
            A list of dictionaries containing information about Django models
        """
        result = []
        
        try:
            app_config = apps.get_app_config(app_label)
        except LookupError:
            raise ValueError(f"App '{app_label}' does not exist")
            
        for model in app_config.get_models():
            model_info = {
                "name": model.__name__,
                "verbose_name": str(model._meta.verbose_name),
                "verbose_name_plural": str(model._meta.verbose_name_plural),
                "fields_count": len(model._meta.fields),
                "has_admin": model.__name__ in [m.__name__ for m in app_config.get_models() if hasattr(m, 'Admin')],
            }
            
            result.append(model_info)
            
        return result


class DjangoModelInfoGuard(DjangoGuard):
    """Function guard for getting information about a Django model."""
    
    def __init__(self):
        """Initialize the Django model info guard."""
        super().__init__(self._get_model_info)
    
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if it's safe to get information about a Django model.
        
        Args:
            **kwargs: The arguments to the function
            
        Returns:
            True (getting model info is always safe)
        """
        return True
    
    def _get_model_info(self, app_label: str, model_name: str, pk: Optional[Any] = None) -> Dict[str, Any]:
        """Get information about a Django model.
        
        Args:
            app_label: The label of the Django app containing the model
            model_name: The name of the model
            pk: The primary key of the model instance to get information about
            
        Returns:
            A dictionary containing information about the Django model or instance
        """
        model_class = self._get_model_class(app_label, model_name)
        
        if pk is not None:
            # Get information about a specific instance
            instance = self._get_model_instance(model_class, pk)
            
            result = {
                "model": model_name,
                "app_label": app_label,
                "pk": pk,
                "data": self._model_to_dict(instance),
            }
        else:
            # Get information about the model schema
            result = {
                "model": model_name,
                "app_label": app_label,
                "verbose_name": str(model_class._meta.verbose_name),
                "verbose_name_plural": str(model_class._meta.verbose_name_plural),
                "fields": self._get_model_fields(model_class),
                "primary_key_field": model_class._meta.pk.name,
            }
            
        return result


class DjangoModelCreateGuard(DjangoGuard):
    """Function guard for creating Django model instances."""
    
    def __init__(self):
        """Initialize the Django model create guard."""
        super().__init__(self._create_model_instance)
    
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if it's safe to create a Django model instance.
        
        Args:
            **kwargs: The arguments to the function
            
        Returns:
            True if the operation is safe, False otherwise
        """
        # Basic validation of required parameters
        if not all(k in kwargs for k in ['app_label', 'model_name', 'data']):
            return False
            
        # Ensure data is a dictionary
        if not isinstance(kwargs.get('data', {}), dict):
            return False
            
        return True
    
    def _create_model_instance(self, app_label: str, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Django model instance.
        
        Args:
            app_label: The label of the Django app containing the model
            model_name: The name of the model
            data: The data to use for creating the model instance
            
        Returns:
            A dictionary containing information about the created model instance
        """
        model_class = self._get_model_class(app_label, model_name)
        
        # Create the instance
        try:
            instance = model_class(**data)
            instance.full_clean()  # Validate the instance
            instance.save()
            
            return {
                "success": True,
                "model": model_name,
                "app_label": app_label,
                "pk": instance.pk,
                "data": self._model_to_dict(instance),
            }
        except ValidationError as e:
            return {
                "success": False,
                "model": model_name,
                "app_label": app_label,
                "errors": dict(e),
            }
        except Exception as e:
            return {
                "success": False,
                "model": model_name,
                "app_label": app_label,
                "error": str(e),
            }


class DjangoModelUpdateGuard(DjangoGuard):
    """Function guard for updating Django model instances."""
    
    def __init__(self):
        """Initialize the Django model update guard."""
        super().__init__(self._update_model_instance)
    
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if it's safe to update a Django model instance.
        
        Args:
            **kwargs: The arguments to the function
            
        Returns:
            True if the operation is safe, False otherwise
        """
        # Basic validation of required parameters
        if not all(k in kwargs for k in ['app_label', 'model_name', 'pk', 'data']):
            return False
            
        # Ensure data is a dictionary
        if not isinstance(kwargs.get('data', {}), dict):
            return False
            
        return True
    
    def _update_model_instance(self, app_label: str, model_name: str, pk: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a Django model instance.
        
        Args:
            app_label: The label of the Django app containing the model
            model_name: The name of the model
            pk: The primary key of the model instance to update
            data: The data to use for updating the model instance
            
        Returns:
            A dictionary containing information about the updated model instance
        """
        model_class = self._get_model_class(app_label, model_name)
        
        # Get the instance
        try:
            instance = self._get_model_instance(model_class, pk)
            
            # Update the instance
            for key, value in data.items():
                setattr(instance, key, value)
                
            instance.full_clean()  # Validate the instance
            instance.save()
            
            return {
                "success": True,
                "model": model_name,
                "app_label": app_label,
                "pk": instance.pk,
                "data": self._model_to_dict(instance),
            }
        except ValidationError as e:
            return {
                "success": False,
                "model": model_name,
                "app_label": app_label,
                "pk": pk,
                "errors": dict(e),
            }
        except Exception as e:
            return {
                "success": False,
                "model": model_name,
                "app_label": app_label,
                "pk": pk,
                "error": str(e),
            }


class DjangoModelSearchGuard(DjangoGuard):
    """Function guard for searching for Django model instances."""
    
    def __init__(self):
        """Initialize the Django model search guard."""
        super().__init__(self._search_model_instances)
    
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if it's safe to search for Django model instances.
        
        Args:
            **kwargs: The arguments to the function
            
        Returns:
            True if the operation is safe, False otherwise
        """
        # Basic validation of required parameters
        if not all(k in kwargs for k in ['app_label', 'model_name', 'filters']):
            return False
            
        # Ensure filters is a dictionary
        if not isinstance(kwargs.get('filters', {}), dict):
            return False
            
        return True
    
    def _search_model_instances(self, app_label: str, model_name: str, filters: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """Search for Django model instances.
        
        Args:
            app_label: The label of the Django app containing the model
            model_name: The name of the model
            filters: The filters to use for searching for model instances
            limit: The maximum number of results to return
            
        Returns:
            A dictionary containing the search results
        """
        model_class = self._get_model_class(app_label, model_name)
        
        # Build the query
        query = Q()
        invalid_fields = []
        
        for field_name, value in filters.items():
            try:
                # Check if the field exists
                model_class._meta.get_field(field_name)
                
                # Add the filter to the query
                query &= Q(**{field_name: value})
            except FieldDoesNotExist:
                invalid_fields.append(field_name)
                
        # Execute the query
        try:
            instances = model_class.objects.filter(query)[:limit]
            
            results = []
            for instance in instances:
                results.append({
                    "pk": instance.pk,
                    "data": self._model_to_dict(instance),
                })
                
            return {
                "success": True,
                "model": model_name,
                "app_label": app_label,
                "count": len(results),
                "results": results,
                "invalid_fields": invalid_fields,
            }
        except Exception as e:
            return {
                "success": False,
                "model": model_name,
                "app_label": app_label,
                "error": str(e),
                "invalid_fields": invalid_fields,
            }


def create_django_app_list_tool(
    name: str = "list_django_apps",
    description: str = "List available Django apps/modules."
) -> ToolParams:
    """Create a tool for listing Django apps.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for listing Django apps
    """
    guard = DjangoAppListGuard()
    
    return ToolParams(
        name=name,
        description=description,
        parameters=DjangoAppListParams.model_json_schema(),
        return_type=List[Dict[str, Any]],
        source=guard
    )


def create_django_model_list_tool(
    name: str = "list_django_models",
    description: str = "List models in a Django app."
) -> ToolParams:
    """Create a tool for listing Django models.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for listing Django models
    """
    guard = DjangoModelListGuard()
    
    return ToolParams(
        name=name,
        description=description,
        parameters=DjangoModelListParams.model_json_schema(),
        return_type=List[Dict[str, Any]],
        source=guard
    )


def create_django_model_info_tool(
    name: str = "get_django_model_info",
    description: str = "Get information about a Django model or model instance."
) -> ToolParams:
    """Create a tool for getting information about Django models.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for getting information about Django models
    """
    guard = DjangoModelInfoGuard()
    
    return ToolParams(
        name=name,
        description=description,
        parameters=DjangoModelInfoParams.model_json_schema(),
        return_type=Dict[str, Any],
        source=guard
    )


def create_django_model_create_tool(
    name: str = "create_django_model_instance",
    description: str = "Create a new Django model instance."
) -> ToolParams:
    """Create a tool for creating Django model instances.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for creating Django model instances
    """
    guard = DjangoModelCreateGuard()
    
    return ToolParams(
        name=name,
        description=description,
        parameters=DjangoModelCreateParams.model_json_schema(),
        return_type=Dict[str, Any],
        source=guard
    )


def create_django_model_update_tool(
    name: str = "update_django_model_instance",
    description: str = "Update an existing Django model instance."
) -> ToolParams:
    """Create a tool for updating Django model instances.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for updating Django model instances
    """
    guard = DjangoModelUpdateGuard()
    
    return ToolParams(
        name=name,
        description=description,
        parameters=DjangoModelUpdateParams.model_json_schema(),
        return_type=Dict[str, Any],
        source=guard
    )


def create_django_model_search_tool(
    name: str = "search_django_model_instances",
    description: str = "Search for Django model instances matching specific criteria."
) -> ToolParams:
    """Create a tool for searching for Django model instances.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for searching for Django model instances
    """
    guard = DjangoModelSearchGuard()
    
    return ToolParams(
        name=name,
        description=description,
        parameters=DjangoModelSearchParams.model_json_schema(),
        return_type=Dict[str, Any],
        source=guard
    )


def create_all_django_tools() -> List[ToolParams]:
    """Create all Django tools.
    
    Returns:
        List[ToolParams]: A list of all Django tools
    """
    return [
        create_django_app_list_tool(),
        create_django_model_list_tool(),
        create_django_model_info_tool(),
        create_django_model_create_tool(),
        create_django_model_update_tool(),
        create_django_model_search_tool(),
    ]
