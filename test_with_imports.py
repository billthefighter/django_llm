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

from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging

# Configure logging to show all messages
logging.basicConfig(level=logging.DEBUG)

# Import directly via importlib
import importlib.util

# Import field_type_mapping.py
spec1 = importlib.util.spec_from_file_location(
    "field_type_mapping",
    "/Users/lucaswhipple/Dropbox/code_projects/django_llm/submodules/pydantic2django/src/pydantic2django/field_type_mapping.py"
)

# Import factory.py
spec2 = importlib.util.spec_from_file_location(
    "factory",
    "/Users/lucaswhipple/Dropbox/code_projects/django_llm/submodules/pydantic2django/src/pydantic2django/factory.py"
)

# Import relationships.py
spec3 = importlib.util.spec_from_file_location(
    "relationships",
    "/Users/lucaswhipple/Dropbox/code_projects/django_llm/submodules/pydantic2django/src/pydantic2django/relationships.py"
)

if all([spec1, spec2, spec3]):
    # Load the modules
    field_type_mapping = importlib.util.module_from_spec(spec1)
    factory = importlib.util.module_from_spec(spec2)
    relationships = importlib.util.module_from_spec(spec3)
    
    # Execute the modules
    spec1.loader.exec_module(field_type_mapping)
    spec2.loader.exec_module(factory)
    spec3.loader.exec_module(relationships)
    
    # Import needed classes
    TypeMapper = field_type_mapping.TypeMapper
    DjangoFieldFactory = factory.DjangoFieldFactory
    FieldConversionResult = factory.FieldConversionResult
    RelationshipConversionAccessor = relationships.RelationshipConversionAccessor
    
    # Define custom problem type
    class CustomType:
        def __init__(self, value):
            self.value = value
    
    # Create test model with problematic fields
    class TestProblemModel(BaseModel):
        # Field with custom type that has no mapping
        custom_field: CustomType 
        
        # Field with callable that can't be stored
        callback: Callable[[str], bool]
        
        # Field with complex type that can't be serialized easily
        complex_dict: Dict[str, List[CustomType]]
        
        # Field that might work but we'll intentionally break it
        normal_field: str = Field(default="test")
    
    # Create field factory for testing
    relationship_accessor = RelationshipConversionAccessor()
    field_factory = DjangoFieldFactory(available_relationships=relationship_accessor)
    
    # Test each field
    print("\n=== Testing Field Conversion Errors ===\n")
    
    for field_name, field_info in TestProblemModel.model_fields.items():
        print(f"\nTesting field: {field_name}")
        print(f"Field type: {field_info.annotation}")
        
        try:
            # Attempt to convert the field
            conversion_result = field_factory.convert_field(
                field_name=field_name, 
                field_info=field_info,
                app_label="test_app"
            )
            print(f"Conversion succeeded! Result: {conversion_result}")
        except Exception as e:
            print(f"Conversion failed with detailed error:")
            print(f"{str(e)}")
    
    print("\n=== Tests Complete ===\n")
else:
    print("Failed to load one or more modules") 