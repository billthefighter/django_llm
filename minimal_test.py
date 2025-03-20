import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'submodules/pydantic2django/src'))

# Configure Django settings before importing any Django module
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

from typing import Set, get_origin, get_args
from pydantic2django.field_type_mapping import TypeMapper

# Test direct set type
print("Testing set types:")
simple_set = set
parameterized_set = Set[str]

# Test simple set
print("\nSimple set:")
mapping = TypeMapper.get_mapping_for_type(simple_set)
print("Mapping for simple set:", mapping)
print("Origin:", get_origin(simple_set))
print("Args:", get_args(simple_set))

# Test parameterized set
print("\nParameterized set:")
mapping = TypeMapper.get_mapping_for_type(parameterized_set)
print("Mapping for Set[str]:", mapping)
print("Origin:", get_origin(parameterized_set))
print("Args:", get_args(parameterized_set))

# Test matches_type method directly
json_field_mapping = next((m for m in TypeMapper.TYPE_MAPPINGS if m.python_type == set), None)
if json_field_mapping:
    print("\nTesting matches_type method:")
    print("Direct match for set:", json_field_mapping.matches_type(set))
    print("Match for Set[str]:", json_field_mapping.matches_type(Set[str])) 