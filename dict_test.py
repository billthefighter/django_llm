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

# Import directly from field_type_mapping.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "field_type_mapping",
    "/Users/lucaswhipple/Dropbox/code_projects/django_llm/submodules/pydantic2django/src/pydantic2django/field_type_mapping.py"
)
if spec:
    field_type_mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(field_type_mapping)
    TypeMappingDefinition = field_type_mapping.TypeMappingDefinition
    TypeMapper = field_type_mapping.TypeMapper

    # Test Dict[str, Any]
    print("Testing Dict[str, Any] type:")
    dict_type = dict
    dict_str_any = Dict[str, Any]

    # Test simple dict
    print("\nSimple dict:")
    mapping = TypeMapper.get_mapping_for_type(dict_type)
    print("Mapping for dict:", mapping)
    print("Origin:", get_origin(dict_type))
    print("Args:", get_args(dict_type))
    if mapping:
        print("Django field:", mapping.django_field.__name__)

    # Test Dict[str, Any]
    print("\nDict[str, Any]:")
    mapping = TypeMapper.get_mapping_for_type(dict_str_any)
    print("Mapping for Dict[str, Any]:", mapping)
    print("Origin:", get_origin(dict_str_any))
    print("Args:", get_args(dict_str_any))
    if mapping:
        print("Django field:", mapping.django_field.__name__)

    # Test matches_type method directly
    json_field_mapping = next((m for m in TypeMapper.TYPE_MAPPINGS if m.python_type == dict), None)
    if json_field_mapping:
        print("\nTesting matches_type method:")
        print("Direct match for dict:", json_field_mapping.matches_type(dict))
        print("Match for Dict[str, Any]:", json_field_mapping.matches_type(Dict[str, Any]))
        
        # Check the origin and args of Dict[str, Any]
        origin = get_origin(dict_str_any)
        args = get_args(dict_str_any)
        print("\nDetailed inspection of Dict[str, Any]:")
        print("Origin type:", type(origin))
        print("Origin value:", origin)
        print("Args:", args)
        
        # Check how the matches_type method handles origin and args
        print("\nMatchType evaluation for Dict[str, Any]:")
        print("Origin in (list, dict, set):", origin in (list, dict, set))
        print("Our python_type:", json_field_mapping.python_type)
        our_origin = get_origin(json_field_mapping.python_type)
        print("Our origin:", our_origin)
        print("Origin == get_origin(json_field_mapping.python_type):", origin == our_origin)
else:
    print("Failed to load field_type_mapping module") 