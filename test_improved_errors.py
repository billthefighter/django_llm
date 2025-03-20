import sys
import os
import django
from django.conf import settings

# Configure Django
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        }
    )
    django.setup()

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Directly load our modified factory.py
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
from django.db import models
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic.config import ConfigDict


# Define a custom type that can't be directly mapped
class CustomType:
    def __init__(self, value):
        self.value = value


# Load modules for testing
print("Creating test models...")

# Create a test model with a problematic field
class TestModel(BaseModel):
    # Allow arbitrary types in this model
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # A field with a type that can't be mapped
    custom_field: CustomType
    
    # Normal field
    name: str = "test"


# Create mock objects for testing
class MockRelationshipAccessor:
    def __init__(self):
        self.available_pydantic_models = set()


# Create simplified versions of required classes
@dataclass
class TypeMappingDefinition:
    python_type: Any
    django_field: Any
    is_relationship: bool = False
    
    def matches_type(self, python_type):
        return self.python_type == python_type


class MockTypeMapper:
    @classmethod
    def get_mapping_for_type(cls, python_type):
        # Mock always returns None for CustomType
        if python_type == CustomType:
            return None
        # Return a field for str type
        if python_type == str:
            return TypeMappingDefinition(str, models.TextField)
        return None
        
    @classmethod
    def get_field_attributes(cls, python_type):
        return {}


@dataclass
class FieldConversionResult:
    field_info: FieldInfo
    field_name: str
    app_label: str
    type_mapping_definition: Optional[TypeMappingDefinition] = None
    field_kwargs: Dict[str, Any] = field(default_factory=dict)
    django_field: Optional[models.Field] = None
    context_field: Optional[FieldInfo] = None
    error_str: Optional[str] = None

    @property
    def rendered_django_field(self):
        if self.django_field and self.type_mapping_definition:
            return models.TextField()
        raise ValueError("Cannot render field")


@dataclass
class DjangoFieldFactory:
    available_relationships: Any
    
    def convert_field(self, field_name, field_info, app_label="test_app"):
        result = FieldConversionResult(
            field_info=field_info,
            field_name=field_name,
            app_label=app_label
        )
        
        try:
            # Simulate handling ID field
            if field_name.lower() == "id":
                result.django_field = models.AutoField(primary_key=True)
            
            # Get field type and attributes
            field_type = field_info.annotation
            result.field_kwargs = {
                "null": not field_info.is_required,
                "blank": not field_info.is_required
            }
            
            # Get mapping from TypeMapper
            result.type_mapping_definition = MockTypeMapper.get_mapping_for_type(field_type)
            
            if not result.type_mapping_definition:
                # Create detailed error message
                detailed_msg = f"Error converting field '{field_name}' of type '{field_info.annotation}': No mapping found"
                detailed_msg += f"\nNo type mapping found in TypeMapper"
                detailed_msg += f"\nField info: {field_info}"
                detailed_msg += f"\nField default: {field_info.default}"
                detailed_msg += f"\nField metadata: {field_info.metadata}"
                
                logging.error(detailed_msg)
                result.error_str = detailed_msg
                
                # Decide if this should be contextual
                if field_type == CustomType:
                    # Here we'd set the context field
                    result.context_field = field_info
                    return result
                
                # Or raise an error
                raise ValueError(detailed_msg)
            
            # If we got here, we have a valid mapping
            result.django_field = models.TextField()
            return result
            
        except Exception as e:
            # Create a detailed error message using our new approach
            detailed_msg = f"Error converting field '{field_name}' of type '{field_info.annotation}': {e}"
            if result.type_mapping_definition:
                detailed_msg += f"\nMapping found but error occurred during processing"
            else:
                detailed_msg += "\nNo type mapping found in TypeMapper"
                
            detailed_msg += f"\nField info: {field_info}"
            detailed_msg += f"\nField default: {field_info.default}"
            detailed_msg += f"\nField metadata: {field_info.metadata}"
            
            logging.error(detailed_msg)
            result.error_str = detailed_msg
            raise ValueError(detailed_msg) from e


# Now test with our fields
field_factory = DjangoFieldFactory(available_relationships=MockRelationshipAccessor())

print("\n=== Testing Field Conversion Errors ===\n")

# Test conversion of each field
for field_name, field_info in TestModel.model_fields.items():
    print(f"\nTesting field: {field_name}")
    print(f"Field type: {field_info.annotation}")
    
    try:
        result = field_factory.convert_field(field_name, field_info)
        print(f"Conversion result: {result}")
        
        if result.context_field:
            print(f"Field was marked as contextual")
        elif result.django_field:
            print(f"Django field created: {result.django_field}")
        
    except Exception as e:
        print(f"Conversion failed with error:")
        print(str(e))

print("\n=== Test Complete ===\n") 