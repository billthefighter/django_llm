"""Utilities for mapping between Pydantic and Django models."""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast
import importlib

from django.db import models

from pydantic2django.base_django_model import Pydantic2DjangoBaseClass

# Type variable for Pydantic2DjangoBaseClass
T = TypeVar('T', bound=Pydantic2DjangoBaseClass)


class ModelMapper:
    """
    Utility class for mapping between Pydantic and Django models.
    
    This class provides methods for finding the appropriate Django model
    for a given Pydantic model and vice versa.
    """
    
    @staticmethod
    def get_django_model_for_pydantic(pydantic_obj: Any) -> Type[Pydantic2DjangoBaseClass]:
        """
        Get the Django model class for a given Pydantic object.
        
        Args:
            pydantic_obj: The Pydantic object
            
        Returns:
            The Django model class
            
        Raises:
            ValueError: If no matching Django model is found
        """
        # Get the object type
        object_type = pydantic_obj.__class__.__name__
        
        # Look for a Django model with the name "Django{object_type}"
        django_model_name = f"Django{object_type}"
        
        # Try to import from django_llm.models.models
        try:
            models_module = importlib.import_module("django_llm.models.models")
            django_model = getattr(models_module, django_model_name, None)
            
            if django_model and issubclass(django_model, Pydantic2DjangoBaseClass):
                return django_model
        except (ImportError, AttributeError):
            pass
        
        # Try to find the model in all installed apps
        from django.apps import apps
        for model in apps.get_models():
            if model.__name__ == django_model_name and issubclass(model, Pydantic2DjangoBaseClass):
                return cast(Type[Pydantic2DjangoBaseClass], model)
        
        raise ValueError(f"No Django model found for Pydantic model {object_type}")
    
    @staticmethod
    def get_pydantic_class_for_django(django_obj: Pydantic2DjangoBaseClass) -> Type[Any]:
        """
        Get the Pydantic class for a given Django object.
        
        Args:
            django_obj: The Django object
            
        Returns:
            The Pydantic class
            
        Raises:
            ValueError: If no matching Pydantic class is found
        """
        # Get the object type - ensure it's a string
        object_type = str(django_obj.object_type)
        
        # Get the module path
        try:
            module_path = django_obj._get_module_path()
        except NotImplementedError:
            raise ValueError(f"Django model {django_obj.__class__.__name__} does not implement _get_module_path")
        
        # Import the module
        try:
            module = importlib.import_module(module_path)
            pydantic_class = getattr(module, object_type, None)
            
            if pydantic_class:
                return pydantic_class
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Error importing Pydantic class {object_type} from {module_path}: {e}")
        
        raise ValueError(f"No Pydantic class found for Django model {django_obj.__class__.__name__}")
    
    @staticmethod
    def find_or_create_django_object(
        pydantic_obj: Any, 
        django_model: Optional[Type[Pydantic2DjangoBaseClass]] = None,
        unique_fields: Optional[List[str]] = None
    ) -> Pydantic2DjangoBaseClass:
        """
        Find an existing Django object for a Pydantic object or create a new one.
        
        Args:
            pydantic_obj: The Pydantic object
            django_model: Optional Django model class (will be inferred if not provided)
            unique_fields: Optional list of fields to use for uniqueness check
            
        Returns:
            The Django object
        """
        # Get the Django model if not provided
        if django_model is None:
            django_model = ModelMapper.get_django_model_for_pydantic(pydantic_obj)
        
        # If unique fields are provided, try to find an existing object
        if unique_fields and django_model:
            query_params = {}
            data = pydantic_obj.model_dump()
            
            for field in unique_fields:
                if field in data:
                    query_params[field] = data[field]
            
            if query_params:
                try:
                    # Use type: ignore to bypass type checking for Django model methods
                    return django_model.objects.get(**query_params)  # type: ignore
                except django_model.DoesNotExist:  # type: ignore
                    pass
        
        # Create a new object
        if django_model:
            return django_model.from_pydantic(pydantic_obj)  # type: ignore
        else:
            raise ValueError("Django model is required to create an object")
    
    @staticmethod
    def sync_django_object(django_obj: Pydantic2DjangoBaseClass, pydantic_obj: Any) -> None:
        """
        Synchronize a Django object with a Pydantic object.
        
        Args:
            django_obj: The Django object to update
            pydantic_obj: The Pydantic object with the new data
        """
        # Update the Django object
        django_obj.update_from_pydantic(pydantic_obj)  # type: ignore
        
        # Sync database fields
        django_obj.sync_db_fields_from_data()  # type: ignore