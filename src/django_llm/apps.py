"""
Django LLM app configuration.
"""
import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class DjangoLLMConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'django_llm'
    verbose_name = 'Django LLM'
    
    def ready(self):
        """
        Initialize app configuration.
        Model discovery is deferred until migrations to avoid registration issues.
        """
        logger.info("Initializing Django-LLM app...")
        
        # We no longer call discover_models here
        # Model discovery will happen during migrations
        pass 