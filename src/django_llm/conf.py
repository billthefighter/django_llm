from django.conf import settings

DEFAULTS = {
    'STORAGE_BACKEND': 'django_llm.storage.DatabaseStorage',
    'TOKEN_TRACKING': True,
    'COST_TRACKING': True,
    
    # Additional suggested settings
    'DEFAULT_MODEL': 'gpt-3.5-turbo',  # Default LLM model to use
    'MAX_TOKENS': 2000,  # Default max tokens for responses
    'TEMPERATURE': 0.7,  # Default temperature for responses
    'API_KEY_SETTING': None,  # Which Django setting contains the API key
    'CACHE_RESPONSES': False,  # Whether to cache LLM responses
    'CACHE_TTL': 3600,  # How long to cache responses (in seconds)
    'LOG_LEVEL': 'INFO',  # Logging level for LLM operations
}

class Settings:
    """
    Settings wrapper for django-llm that handles defaults and prefixed settings
    """
    def __getattr__(self, name):
        if name not in DEFAULTS:
            raise AttributeError(f"django-llm setting '{name}' does not exist")
            
        return getattr(settings, f'DJANGO_LLM_{name}', DEFAULTS[name])

# Create a settings object to use throughout the package
llm_settings = Settings() 