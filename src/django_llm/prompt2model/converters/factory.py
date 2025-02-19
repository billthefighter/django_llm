from typing import List, Type

from django.db import models
from llmaestro.llm.interfaces.base import BaseLLMInterface

from ..model_inspector import inspect_model
from .base import BaseModelConverter, ComplexModelConverter, SimpleModelConverter


class ModelConverterFactory:
    """Factory for creating appropriate model converters."""
    
    @staticmethod
    def has_relationships(model_class: Type[models.Model]) -> bool:
        """Check if model has any relationships."""
        model_info = inspect_model(model_class)
        return bool(model_info.get('relationships'))
    
    @classmethod
    def create_converter(
        cls,
        target_models: List[Type[models.Model]],
        llm_interface: BaseLLMInterface
    ) -> BaseModelConverter:
        """
        Create appropriate converter based on model complexity.
        
        Args:
            target_models: List of model classes to convert
            llm_interface: LLM interface for conversion
        
        Returns:
            Appropriate model converter instance
        """
        # Use simple converter for single model without relationships
        if len(target_models) == 1 and not cls.has_relationships(target_models[0]):
            return SimpleModelConverter(llm_interface)
        
        # Use complex converter for multiple models or models with relationships
        return ComplexModelConverter(llm_interface) 