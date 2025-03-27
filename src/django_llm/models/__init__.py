try:
    from .models import __all__
    if not __all__:
        pass
except (ImportError, NameError):
    pass