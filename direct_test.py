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

from typing import Set, get_origin, get_args
from django.db import models

# Import directly from field_type_mapping.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "field_type_mapping",
    "/Users/lucaswhipple/Dropbox/code_projects/django_llm/submodules/pydantic2django/src/pydantic2django/field_type_mapping.py"
)
field_type_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(field_type_mapping)
TypeMappingDefinition = field_type_mapping.TypeMappingDefinition
TypeMapper = field_type_mapping.TypeMapper

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
if mapping:
    print("Django field:", mapping.django_field.__name__)

# Test parameterized set
print("\nParameterized set:")
mapping = TypeMapper.get_mapping_for_type(parameterized_set)
print("Mapping for Set[str]:", mapping)
print("Origin:", get_origin(parameterized_set))
print("Args:", get_args(parameterized_set))
if mapping:
    print("Django field:", mapping.django_field.__name__)

# Test matches_type method directly
json_field_mapping = next((m for m in TypeMapper.TYPE_MAPPINGS if m.python_type == set), None)
if json_field_mapping:
    print("\nTesting matches_type method:")
    print("Direct match for set:", json_field_mapping.matches_type(set))
    print("Match for Set[str]:", json_field_mapping.matches_type(Set[str]))
    
    # Check the origin and args of the parameterized set
    origin = get_origin(parameterized_set)
    args = get_args(parameterized_set)
    print("\nDetailed inspection of Set[str]:")
    print("Origin type:", type(origin))
    print("Origin value:", origin)
    print("Args:", args)
    
    # Check how the matches_type method handles origin and args
    print("\nMatchType evaluation for Set[str]:")
    print("Origin in (list, dict, set):", origin in (list, dict, set))
    print("Our python_type:", json_field_mapping.python_type)
    our_origin = get_origin(json_field_mapping.python_type)
    print("Our origin:", our_origin)
    print("Origin == get_origin(json_field_mapping.python_type):", origin == our_origin) 