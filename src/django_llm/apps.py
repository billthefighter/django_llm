"""
Django LLM app configuration.
"""
import logging
from django.apps import AppConfig
from django.db.models import signals


logger = logging.getLogger(__name__)


class DjangoLLMConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'django_llm'
    verbose_name = 'Django LLM'
    
    def ready(self):
        """
        Initialize app configuration and discover models.
        Actual model registration happens during migration operations.
        """
        from .models.model_discovery import discover_models
        
        logger.info("Initializing Django-LLM app...")
        
        # Just discover models during app initialization
        # This doesn't create any Django models or access the database
        discover_models(self.name) 