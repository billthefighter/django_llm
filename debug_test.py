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

# Add the submodules path to sys.path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'submodules/pydantic2django/src'))

from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging

# Configure logging to show all messages
logging.basicConfig(level=logging.DEBUG)

# Import the modules we need to test
from pydantic2django.factory import DjangoFieldFactory, FieldConversionResult
from pydantic2django.field_type_mapping import TypeMapper
from pydantic2django.relationships import RelationshipConversionAccessor

# Create a problematic field type that will fail conversion
class CustomType:
    def __init__(self, value):
        self.value = value

# Create a test model with various problematic fields
class TestProblemModel(BaseModel):
    # Field with custom type that has no mapping
    custom_field: CustomType 
    
    # Field with callable that can't be stored
    callback: Callable[[str], bool]
    
    # Field with complex type that can't be serialized easily
    complex_dict: Dict[str, List[CustomType]]
    
    # Field that might work but we'll intentionally break it
    normal_field: str = Field(default="test")

# Create a field factory for testing
relationship_accessor = RelationshipConversionAccessor()
field_factory = DjangoFieldFactory(available_relationships=relationship_accessor)

# Test each field and catch the error messages
fields_to_test = TestProblemModel.model_fields.items()

print("\n=== Testing Field Conversion Errors ===\n")

for field_name, field_info in fields_to_test:
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