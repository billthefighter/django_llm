import sys
import os
import django
from django.conf import settings

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

from typing import Dict, Any, get_origin, get_args
from django.db import models
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

# Import directly from field_type_mapping.py and factory.py
import importlib.util
spec1 = importlib.util.spec_from_file_location(
    "field_type_mapping",
    "/Users/lucaswhipple/Dropbox/code_projects/django_llm/submodules/pydantic2django/src/pydantic2django/field_type_mapping.py"
)
spec2 = importlib.util.spec_from_file_location(
    "factory",
    "/Users/lucaswhipple/Dropbox/code_projects/django_llm/submodules/pydantic2django/src/pydantic2django/factory.py"
)

if spec1 and spec2:
    field_type_mapping = importlib.util.module_from_spec(spec1)
    factory = importlib.util.module_from_spec(spec2)
    
    spec1.loader.exec_module(field_type_mapping)
    spec2.loader.exec_module(factory)
    
    # To test where context_field is set
    print("Analyzing FieldConversionResult class...")
    
    # Define a test class with Dict[str, Any]
    class TestModel(BaseModel):
        variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Check the field info
    field_info = TestModel.model_fields["variables"]
    print(f"Field info: {field_info}")
    print(f"Field annotation: {field_info.annotation}")
    print(f"Field annotation repr: {repr(field_info.annotation)}")
    
    # Our goal is to find out why the context_field is getting set
    # Even though the code can now map Dict[str, Any] to JSONField
    print("\nChecking TypeMapper for Dict[str, Any]...")
    TypeMapper = field_type_mapping.TypeMapper
    mapping = TypeMapper.get_mapping_for_type(field_info.annotation)
    print(f"Mapping found: {mapping}")
    
    print("\nFinding out where context_field is getting set...")
    
    # Create FieldConversionResult to inspect
    FieldConversionResult = factory.FieldConversionResult
    
    # Example result
    result = FieldConversionResult(
        field_info=field_info,
        field_name="variables",
        app_label="test_app",
        type_mapping_definition=mapping
    )
    
    # Check if context_field is set somewhere
    print(f"Initial context_field: {result.context_field}")
    
    # Let's see how the result is processed in convert_field
    print("\nDebug points in convert_field:")
    print("1. After getting mapping from TypeMapper:")
    print(f"type_mapping_definition: {mapping}")
    
    # The critical point where context_field might be set
    if not mapping:
        print("Would enter block: 'if not result.type_mapping_definition:'")
        print("This would log: 'Could not map variables of type typing.Dict[str, typing.Any] to a Django field, must be contextual'")
        print("However, context_field is not set here, which suggests it might be set elsewhere or by another condition")
    else:
        print("Would NOT enter the warning block since mapping was found")
        print("This suggests the context_field might be set in another part of the code")
        
    # Check rendered_django_field
    if mapping:
        try:
            django_field = mapping.get_django_field({})
            print(f"\nDjango field created: {django_field}")
        except Exception as e:
            print(f"Error creating Django field: {e}")
else:
    print("Failed to load modules") 