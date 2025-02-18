from typing import Type, TypeVar, Optional, Dict, Any
from django.db import models
from pydantic import BaseModel
from llm_orchestrator.src.core.models import (
    Task, SubTask, TaskStatus, TokenUsage, 
    ContextMetrics, AgentConfig
)

T = TypeVar('T', bound=BaseModel)

class LLMaestroModel(models.Model):
    """
    Abstract base class for Django models that sync with LLMaestro Pydantic models.
    """
    llmaestro_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    def to_llmaestro(self, model_class: Type[T]) -> T:
        """Convert Django model instance to LLMaestro Pydantic model"""
        return model_class(**self.llmaestro_data)

    def update_from_llmaestro(self, llmaestro_instance: BaseModel) -> None:
        """Update Django model from LLMaestro Pydantic model"""
        self.llmaestro_data = llmaestro_instance.dict()
        self.save()

class DjangoTask(LLMaestroModel):
    """Django model representation of LLMaestro Task"""
    task_id = models.CharField(max_length=255, unique=True)
    type = models.CharField(max_length=100)
    status = models.CharField(
        max_length=20,
        choices=[(status.value, status.value) for status in TaskStatus]
    )

    def to_llmaestro(self) -> Task:
        return super().to_llmaestro(Task)

class DjangoSubTask(LLMaestroModel):
    """Django model representation of LLMaestro SubTask"""
    subtask_id = models.CharField(max_length=255, unique=True)
    parent_task = models.ForeignKey(DjangoTask, on_delete=models.CASCADE, related_name='subtasks')
    type = models.CharField(max_length=100)
    status = models.CharField(
        max_length=20,
        choices=[(status.value, status.value) for status in TaskStatus]
    )

    def to_llmaestro(self) -> SubTask:
        return super().to_llmaestro(SubTask)

class DjangoTokenUsage(LLMaestroModel):
    """Django model representation of LLMaestro TokenUsage"""
    task = models.ForeignKey(DjangoTask, on_delete=models.CASCADE, related_name='token_usage')
    
    def to_llmaestro(self) -> TokenUsage:
        return super().to_llmaestro(TokenUsage)

class DjangoAgentConfig(LLMaestroModel):
    """Django model representation of LLMaestro AgentConfig"""
    name = models.CharField(max_length=100, unique=True)
    is_active = models.BooleanField(default=True)
    
    def to_llmaestro(self) -> AgentConfig:
        return super().to_llmaestro(AgentConfig) 