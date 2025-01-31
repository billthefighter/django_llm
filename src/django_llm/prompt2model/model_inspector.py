from django.db import models
from typing import Dict, Any, Type, List
import inspect

def inspect_model(model_class: Type[models.Model]) -> Dict[str, Any]:
    """
    Introspect a Django model and return a dictionary of its fields and their properties.
    """
    field_info = {}
    
    for field in model_class._meta.fields:
        field_data = {
            'type': field.__class__.__name__,
            'required': not field.null and not field.blank,
            'help_text': str(field.help_text) if field.help_text else None,
        }
        
        # Add specific field properties
        if isinstance(field, models.CharField):
            field_data['max_length'] = field.max_length
        elif isinstance(field, models.DecimalField):
            field_data['max_digits'] = field.max_digits
            field_data['decimal_places'] = field.decimal_places
        
        # Add choices if they exist
        if field.choices:
            field_data['choices'] = dict(field.choices)
            
        # Add default if it exists
        if field.default != models.NOT_PROVIDED:
            # Handle callable defaults
            if callable(field.default):
                field_data['default'] = 'callable'
            else:
                field_data['default'] = field.default
                
        field_info[field.name] = field_data
    
    # Add model metadata
    model_info = {
        'name': model_class.__name__,
        'fields': field_info,
        'meta': {
            'verbose_name': model_class._meta.verbose_name,
            'verbose_name_plural': model_class._meta.verbose_name_plural,
        }
    }
    
    return model_info 