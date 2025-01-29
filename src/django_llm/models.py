from django.db import models
from django.contrib.postgres.fields import JSONField
from typing import Optional, Dict, Any
import uuid

class LLMProvider(models.Model):
    """Model to track LLM provider configurations based on schema"""
    name = models.CharField(
        max_length=50,
        unique=True,
        help_text="Provider identifier (e.g., 'anthropic')"
    )
    model = models.CharField(
        max_length=100,
        help_text="Model identifier (e.g., 'claude-3-sonnet-20240229')"
    )
    api_key = models.CharField(
        max_length=255,
        help_text="API key for provider authentication"
    )
    max_tokens = models.IntegerField(
        default=1024,
        help_text="Maximum tokens for response generation"
    )
    temperature = models.FloatField(
        default=0.7,
        help_text="Sampling temperature for response generation"
    )
    config = JSONField(
        default=dict,
        help_text="Additional provider-specific configuration"
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this provider configuration is active"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "LLM Provider"
        verbose_name_plural = "LLM Providers"
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['is_active']),
        ]

    @classmethod
    def get_available_models(cls, provider_name: str) -> list:
        """Get available models for a provider based on schema"""
        PROVIDER_MODELS = {
            'anthropic': [
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240229'
            ]
        }
        return PROVIDER_MODELS.get(provider_name, [])

    def get_config_dict(self) -> Dict[str, Any]:
        """Convert provider settings to LLM Orchestrator config format"""
        return {
            "llm": {
                "provider": self.name,
                "model": self.model,
                "api_key": self.api_key,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **self.config
            }
        }

    def validate_model(self) -> bool:
        """Validate that the selected model is available for the provider"""
        available_models = self.get_available_models(self.name)
        return self.model in available_models

    def save(self, *args, **kwargs):
        """Validate model before saving"""
        if not self.validate_model():
            raise ValueError(
                f"Invalid model '{self.model}' for provider '{self.name}'. "
                f"Available models: {', '.join(self.get_available_models(self.name))}"
            )
        super().save(*args, **kwargs)

class ChainExecution(models.Model):
    """Tracks a single execution of a chain"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chain_type = models.CharField(max_length=100)  # SequentialChain, ParallelChain, etc.
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('running', 'Running'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        default='running'
    )
    error_message = models.TextField(null=True, blank=True)
    metadata = JSONField(default=dict, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['chain_type']),
            models.Index(fields=['status']),
            models.Index(fields=['started_at']),
        ]

class ChainStep(models.Model):
    """Represents a single step within a chain execution"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chain_execution = models.ForeignKey(
        ChainExecution,
        on_delete=models.CASCADE,
        related_name='steps'
    )
    step_type = models.CharField(max_length=100)
    order = models.IntegerField()
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    input_data = JSONField(default=dict)
    output_data = JSONField(default=dict, null=True)
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['order']
        indexes = [
            models.Index(fields=['step_type']),
            models.Index(fields=['started_at']),
        ]

class TokenUsageLog(models.Model):
    """Tracks token usage for LLM requests"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chain_step = models.ForeignKey(
        'ChainStep',
        on_delete=models.CASCADE,
        related_name='token_usage'
    )
    provider = models.ForeignKey(
        LLMProvider,
        on_delete=models.PROTECT,
        related_name='usage_logs'
    )
    prompt_tokens = models.IntegerField()
    completion_tokens = models.IntegerField()
    total_tokens = models.IntegerField()
    estimated_cost = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True
    )
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['timestamp']),
        ]

class StoredArtifact(models.Model):
    """Stores artifacts generated during chain execution"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chain_execution = models.ForeignKey(
        ChainExecution,
        on_delete=models.CASCADE,
        related_name='artifacts'
    )
    step = models.ForeignKey(
        ChainStep,
        on_delete=models.CASCADE,
        related_name='artifacts',
        null=True
    )
    name = models.CharField(max_length=255)
    data = JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['created_at']),
        ] 