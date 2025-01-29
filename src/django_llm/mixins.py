from typing import Any, Optional
from .interfaces import DjangoChainTracker

class DjangoChainMixin:
    """Mixin to add Django storage capabilities to chains"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = DjangoChainTracker(
            chain_type=self.__class__.__name__,
            metadata={
                'provider': self.llm.config.provider,
                'model': self.llm.config.model
            }
        )
        # Replace the storage manager with Django implementation
        self.storage = self.tracker.storage
        
    async def execute(self, **kwargs: Any) -> Any:
        """Execute with Django tracking"""
        try:
            result = await super().execute(**kwargs)
            self.tracker.complete_chain()
            return result
        except Exception as e:
            self.tracker.complete_chain(str(e))
            raise 