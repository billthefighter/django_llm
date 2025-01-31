from typing import Any, Dict, List, Type, Union
from django.db import models, transaction
from django.core.exceptions import ValidationError
from datetime import datetime
from .model_inspector import inspect_model
from llm.interfaces.base import BaseLLMInterface
import json

class ModelConversionError(Exception):
    pass

async def convert_to_models(
    data: Any,
    target_models: List[Type[models.Model]],
    llm_interface: BaseLLMInterface
) -> List[models.Model]:
    """
    Convert input data to Django models using LLM assistance.
    
    Args:
        data: Input data to convert
        target_models: List of Django model classes to create
        llm_interface: LLM interface for data conversion
    
    Returns:
        List of created Django model instances
    """
    try:
        # Generate model definitions for prompt
        model_definitions = {
            model.__name__: inspect_model(model)
            for model in target_models
        }
        
        # Prepare the system prompt with model definitions
        system_prompt = f"""
        You are converting data into Django models with the following definitions:
        {json.dumps(model_definitions, indent=2)}
        
        Please convert the input data into valid model dictionaries.
        """
        
        # Get LLM response
        response = await llm_interface.process(
            input_data=data,
            system_prompt=system_prompt
        )
        
        # Parse LLM response
        try:
            conversion_data = json.loads(response.content)
        except json.JSONDecodeError as e:
            raise ModelConversionError(f"Failed to parse LLM response: {e}")
        
        created_models = []
        
        # Create models within transaction
        with transaction.atomic():
            for model_data in conversion_data['model_data']:
                model_class = next(
                    m for m in target_models 
                    if m.__name__ == model_data['model']
                )
                
                # Create main model
                instance = model_class(**model_data['fields'])
                instance.full_clean()
                instance.save()
                
                # Handle relationships
                for relation in model_data.get('relationships', []):
                    related_model = next(
                        m for m in target_models 
                        if m.__name__ == relation['model']
                    )
                    related_instance = related_model.objects.create(
                        **relation['fields']
                    )
                    # Add to appropriate relationship
                    # This would need to be enhanced based on relationship type
                
                created_models.append(instance)
        
        return created_models
    
    except Exception as e:
        raise ModelConversionError(f"Model conversion failed: {str(e)}") 