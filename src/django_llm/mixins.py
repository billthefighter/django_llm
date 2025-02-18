from typing import Any, Optional, Type
from .interfaces import DjangoChainTracker
from django.db import models
from pydantic import BaseModel

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

class LLMaestroMixin:
    """
    Mixin to handle conversion between Django and LLMaestro models.
    """
    
    def create_from_llmaestro(self, llmaestro_instance: BaseModel) -> 'LLMaestroModel':
        """Create a new Django model instance from a LLMaestro instance"""
        instance = self.model(
            llmaestro_data=llmaestro_instance.dict()
        )
        
        # Map common fields
        for field in ['id', 'type', 'status']:
            if hasattr(llmaestro_instance, field):
                setattr(instance, field, getattr(llmaestro_instance, field))
                
        instance.save()
        return instance

    def update_from_llmaestro(self, instance: 'LLMaestroModel', 
                             llmaestro_instance: BaseModel) -> 'LLMaestroModel':
        """Update an existing Django model instance from a LLMaestro instance"""
        instance.llmaestro_data = llmaestro_instance.dict()
        
        # Map common fields
        for field in ['type', 'status']:
            if hasattr(llmaestro_instance, field):
                setattr(instance, field, getattr(llmaestro_instance, field))
                
        instance.save()
        return instance 