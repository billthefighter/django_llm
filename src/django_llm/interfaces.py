from datetime import datetime
from typing import Any, Dict, Optional
import json

from django.db import transaction
from django.utils import timezone

from .models import ChainExecution, ChainStep, TokenUsageLog, StoredArtifact
from submodules.llm_orchestrator.src.utils.storage import StorageManager
from submodules.llm_orchestrator.src.llm.base import LLMResponse, TokenUsage

class DjangoStorageManager(StorageManager):
    """Django implementation of the StorageManager interface"""
    
    def __init__(self, chain_execution_id: str):
        self.chain_execution_id = chain_execution_id
        self._current_step: Optional[ChainStep] = None
        
    @transaction.atomic
    def store_artifact(self, name: str, data: Any) -> None:
        """Store an artifact in the Django database"""
        # Handle Pydantic models
        if hasattr(data, 'model_dump'):
            serialized_data = data.model_dump()
        elif isinstance(data, list) and all(hasattr(item, 'model_dump') for item in data):
            serialized_data = [item.model_dump() for item in data]
        else:
            serialized_data = data
            
        StoredArtifact.objects.create(
            chain_execution_id=self.chain_execution_id,
            step=self._current_step,
            name=name,
            data=serialized_data
        )
        
    def get_artifact(self, name: str) -> Optional[Any]:
        """Retrieve an artifact from the Django database"""
        try:
            artifact = StoredArtifact.objects.get(
                chain_execution_id=self.chain_execution_id,
                name=name
            )
            return artifact.data
        except StoredArtifact.DoesNotExist:
            return None
            
    def set_current_step(self, step: ChainStep) -> None:
        """Set the current step context"""
        self._current_step = step

class DjangoChainTracker:
    """Tracks chain execution in Django"""
    
    def __init__(self, chain_type: str, metadata: Optional[Dict[str, Any]] = None):
        self.chain_execution = ChainExecution.objects.create(
            chain_type=chain_type,
            metadata=metadata or {}
        )
        self.storage = DjangoStorageManager(str(self.chain_execution.id))
        self._step_counter = 0
        
    def start_step(self, step_type: str, input_data: Dict[str, Any]) -> ChainStep:
        """Start tracking a new step"""
        step = ChainStep.objects.create(
            chain_execution=self.chain_execution,
            step_type=step_type,
            order=self._step_counter,
            input_data=input_data
        )
        self._step_counter += 1
        self.storage.set_current_step(step)
        return step
        
    def complete_step(
        self,
        step: ChainStep,
        response: LLMResponse,
        error: Optional[str] = None
    ) -> None:
        """Complete a step with its response"""
        step.completed_at = timezone.now()
        step.output_data = response.model_dump() if hasattr(response, 'model_dump') else response
        step.error_message = error
        step.save()
        
        # Log token usage if available
        if response.token_usage:
            TokenUsageLog.objects.create(
                chain_step=step,
                prompt_tokens=response.token_usage.prompt_tokens,
                completion_tokens=response.token_usage.completion_tokens,
                total_tokens=response.token_usage.total_tokens,
                estimated_cost=response.token_usage.estimated_cost,
                provider=self.chain_execution.metadata.get('provider', 'unknown'),
                model_name=response.metadata.get('model', 'unknown')
            )
            
    def complete_chain(self, error: Optional[str] = None) -> None:
        """Mark the chain execution as complete"""
        self.chain_execution.completed_at = timezone.now()
        self.chain_execution.status = 'failed' if error else 'completed'
        self.chain_execution.error_message = error
        self.chain_execution.save() 