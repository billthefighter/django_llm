"""
Django admin configuration for Django-LLM models.
"""
from django.contrib import admin


# Dynamic model admin class for discovered models
class DynamicModelAdmin(admin.ModelAdmin):
    def __init__(self, model, admin_site):
        self.list_display = [field.name for field in model._meta.fields]
        self.search_fields = ['name'] if 'name' in [f.name for f in model._meta.fields] else ['id']
        self.list_filter = []
        
        # Add timestamp fields to list_filter if they exist
        field_names = [f.name for f in model._meta.fields]
        if 'created_at' in field_names:
            self.list_filter.append('created_at')
        if 'updated_at' in field_names:
            self.list_filter.append('updated_at')
        if 'timestamp' in field_names:
            self.list_filter.append('timestamp')
            
        # Add status and type fields to list_filter if they exist
        if 'status' in field_names:
            self.list_filter.append('status')
        if 'type' in field_names:
            self.list_filter.append('type')
            
        super().__init__(model, admin_site) 