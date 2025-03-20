from pydantic import BaseModel
from typing import Set, get_origin, get_args
import inspect
import sys
import os
sys.path.append('/Users/lucaswhipple/Dropbox/code_projects/django_llm')

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

from submodules.pydantic2django.src.pydantic2django.field_type_mapping import TypeMapper

class Test(BaseModel):
    my_set: Set[str]
    simple_set: set

print("Annotation from model_fields:", repr(Test.model_fields['my_set'].annotation))
print("Origin:", get_origin(Test.model_fields['my_set'].annotation))
print("Args:", get_args(Test.model_fields['my_set'].annotation))
print("\nSimple set:")
print("Annotation from model_fields:", repr(Test.model_fields['simple_set'].annotation))
print("Origin:", get_origin(Test.model_fields['simple_set'].annotation))
print("Args:", get_args(Test.model_fields['simple_set'].annotation))

print("\nTypeMapper handling:")
set_type = Set[str]
mapping = TypeMapper.get_mapping_for_type(set_type)
print("Mapping for Set[str]:", mapping)
print("Is set[str] supported:", TypeMapper.is_type_supported(set_type))

simple_set_type = set
mapping = TypeMapper.get_mapping_for_type(simple_set_type)
print("\nMapping for simple set:", mapping)
print("Is simple set supported:", TypeMapper.is_type_supported(simple_set_type)) 