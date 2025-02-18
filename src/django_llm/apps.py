"""
Django LLM app configuration.
"""
from django.apps import AppConfig


class DjangoLLMConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'django_llm'
    verbose_name = 'Django LLM'
    
    def ready(self):
        """
        Register models and admin classes when the app is ready.
        This ensures models are only registered once and in the correct order.
        """
        from .models import setup_dynamic_models
        
        # Setup models, including discovery, registration and admin
        setup_dynamic_models(app_label=self.name) 