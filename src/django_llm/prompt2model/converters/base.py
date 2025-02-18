from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Optional, Tuple
from django.db import models
from django.db import transaction
from ..model_inspector import inspect_model
from llmaestro.llm.interfaces.base import BaseLLMInterface
from django.core.exceptions import ValidationError
import yaml
import os
import json

class ModelConversionError(Exception):
    pass

class BaseModelConverter(ABC):
    """Base class for model converters."""
    
    MAX_RETRIES = 3  # Maximum number of retry attempts
    
    def __init__(self, llm_interface: BaseLLMInterface):
        self.llm_interface = llm_interface
        self.prompt_template = self._load_prompt_template()
    
    @property
    @abstractmethod
    def prompt_file(self) -> str:
        """Return the path to the prompt template file."""
        pass
    
    def _load_prompt_template(self) -> Dict[str, Any]:
        """Load the prompt template from YAML file."""
        prompt_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "prompts", 
            self.prompt_file
        )
        with open(prompt_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _prepare_system_prompt(
        self, 
        model_definitions: Dict[str, Any]
    ) -> str:
        """Prepare system prompt with model definitions."""
        return self.prompt_template['system_prompt'].format(
            model_definitions=json.dumps(model_definitions, indent=2)
        )
    
    async def _attempt_conversion(
        self,
        data: Any,
        model_class: Type[models.Model],
        conversion_data: Dict[str, Any],
        previous_errors: Optional[List[str]] = None
    ) -> Tuple[Optional[models.Model], Optional[List[str]]]:
        """
        Attempt to create a model instance with validation.
        
        Returns:
            Tuple of (created instance or None, list of validation errors or None)
        """
        try:
            instance = model_class(**conversion_data['fields'])
            instance.full_clean()
            return instance, None
        except ValidationError as e:
            error_messages = []
            for field, errors in e.message_dict.items():
                error_messages.append(f"Field '{field}': {', '.join(errors)}")
            return None, error_messages
        except Exception as e:
            return None, [str(e)]

    def _create_retry_prompt(
        self,
        original_prompt: str,
        validation_errors: List[str],
        expected_format: Dict[str, Any],
        attempt: int
    ) -> str:
        """Create a prompt for retry attempts."""
        retry_prompt = f"""
Previous attempt failed with the following validation errors:
{chr(10).join(f'- {error}' for error in validation_errors)}

Expected response format:
{json.dumps(expected_format, indent=2)}

Please correct these issues and provide a new response that:
1. Addresses all validation errors
2. Matches the expected format exactly
3. Contains valid data for all required fields

This is attempt {attempt + 1} of {self.MAX_RETRIES}.

Original prompt:
{original_prompt}
"""
        return retry_prompt

    async def _process_with_retries(
        self,
        data: Any,
        model_class: Type[models.Model],
        system_prompt: str,
        expected_format: Dict[str, Any]
    ) -> models.Model:
        """Process conversion with retries on validation failure."""
        previous_errors = None
        
        for attempt in range(self.MAX_RETRIES):
            current_prompt = (
                self._create_retry_prompt(system_prompt, previous_errors, expected_format, attempt)
                if previous_errors
                else system_prompt
            )
            
            response = await self.llm_interface.process(
                input_data=data,
                system_prompt=current_prompt
            )
            
            try:
                conversion_data = json.loads(response.content)
                self._validate_response(conversion_data)
                
                instance, validation_errors = await self._attempt_conversion(
                    data, model_class, conversion_data, previous_errors
                )
                
                if instance is not None:
                    return instance
                
                previous_errors = validation_errors
                
            except json.JSONDecodeError as e:
                previous_errors = [f"Invalid JSON response: {str(e)}"]
            except ModelConversionError as e:
                previous_errors = [str(e)]
            
            if attempt == self.MAX_RETRIES - 1:
                error_msg = (
                    f"Failed to convert after {self.MAX_RETRIES} attempts. "
                    f"Last errors: {', '.join(previous_errors)}"
                )
                raise ModelConversionError(error_msg)
    
    @abstractmethod
    async def convert(
        self, 
        data: Any, 
        target_models: List[Type[models.Model]]
    ) -> List[models.Model]:
        """Convert input data to model instances."""
        pass
    
    def _validate_response(self, response: Dict[str, Any]) -> None:
        """Validate LLM response format."""
        if not isinstance(response, dict):
            raise ModelConversionError("Invalid response format")
        
        # Implement specific validation logic in subclasses
        pass

class SimpleModelConverter(BaseModelConverter):
    """Converter for single model without relationships."""
    
    @property
    def prompt_file(self) -> str:
        return "simple_model_converter.yaml"
    
    async def convert(
        self, 
        data: Any, 
        target_models: List[Type[models.Model]]
    ) -> List[models.Model]:
        if len(target_models) != 1:
            raise ModelConversionError("SimpleModelConverter only supports single model conversion")
        
        model_class = target_models[0]
        model_definition = inspect_model(model_class)
        
        system_prompt = self._prepare_system_prompt({"model": model_definition})
        
        expected_format = {
            "fields": {
                field_name: f"<{field_data['type']}>"
                for field_name, field_data in model_definition['fields'].items()
            }
        }
        
        try:
            instance = await self._process_with_retries(
                data=data,
                model_class=model_class,
                system_prompt=system_prompt,
                expected_format=expected_format
            )
            
            with transaction.atomic():
                instance.save()
            
            return [instance]
            
        except Exception as e:
            raise ModelConversionError(f"Model conversion failed: {str(e)}")

class ComplexModelConverter(BaseModelConverter):
    """Converter for models with relationships."""
    
    @property
    def prompt_file(self) -> str:
        return "model_converter.yaml"
    
    async def convert(
        self, 
        data: Any, 
        target_models: List[Type[models.Model]]
    ) -> List[models.Model]:
        model_definitions = {
            model.__name__: inspect_model(model)
            for model in target_models
        }
        
        system_prompt = self._prepare_system_prompt(model_definitions)
        
        expected_format = {
            "model_data": [{
                "model": "<model_name>",
                "fields": {
                    "field_name": "<field_type>"
                },
                "relationships": [{
                    "model": "<related_model_name>",
                    "fields": {
                        "field_name": "<field_type>"
                    }
                }]
            }]
        }
        
        created_instances = []
        
        try:
            # Process each model in the correct order (handling dependencies)
            for model_class in target_models:
                instance = await self._process_with_retries(
                    data=data,
                    model_class=model_class,
                    system_prompt=system_prompt,
                    expected_format=expected_format
                )
                created_instances.append(instance)
            
            return created_instances
            
        except Exception as e:
            # Cleanup any created instances on failure
            for instance in created_instances:
                instance.delete()
            raise ModelConversionError(f"Complex model conversion failed: {str(e)}") 