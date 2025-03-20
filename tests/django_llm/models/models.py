"""
Generated Django models from Pydantic models.
Generated at: 2025-03-19 15:16:56
"""


"""
Imports for generated models and context classes.
"""
# Standard library imports
import uuid
import importlib
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast, TypedDict, Generic
from dataclasses import dataclass, field

# Django and Pydantic imports
from django.db import models
from pydantic import BaseModel

# Pydantic2Django imports
from pydantic2django.base_django_model import Pydantic2DjangoBaseClass, Pydantic2DjangoStorePydanticObject
from pydantic2django.context_storage import ModelContext, FieldContext

# Additional type imports
from typing import Callable, Dict, Optional

# Original Pydantic model imports
from __main__ import BasePrompt, ChainStep, RetryStrategy

# Context class field type imports

# Type variable for model classes
T = TypeVar('T')
# Context classes for models with non-serializable fields
@dataclass
class DjangoChainStepContext(ModelContext):
    """
    Context class for DjangoChainStep.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoChainStep"
    pydantic_class: Type = ChainStep
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="input_transform",
            field_type=Callable,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="output_transform",
            field_type=Callable,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
        
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        input_transform: Optional[Callable],
        output_transform: Optional[Callable]) -> "DjangoChainStepContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            input_transform: Value for input_transform field
            output_transform: Value for output_transform field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("input_transform", input_transform)
        context.set_value("output_transform", output_transform)
        return context


# Generated Django models
"""
Django model for BasePrompt.
"""

class DjangoBasePrompt(Pydantic2DjangoBaseClass[BasePrompt]):
    """
    Django model for BasePrompt.
    """

    prompt = models.TextField(verbose_name='prompt')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_baseprompt"
        app_label = "django_llm"
        verbose_name = """BasePrompt"""
        verbose_name_plural = """BasePrompts"""
        abstract = False


"""
Django model for ChainStep.
"""

class DjangoChainStep(Pydantic2DjangoBaseClass[ChainStep]):
    """
    Django model for ChainStep.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - input_transform: Optional[Any]
        - output_transform: Optional[LLMResponse]
    """

    prompt = models.ForeignKey(verbose_name='prompt', to='django_llm.BasePrompt', on_delete=models.CASCADE)
    retry_strategy = models.ForeignKey(verbose_name='retry_strategy', to='django_llm.DjangoRetryStrategy', on_delete=models.CASCADE)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chainstep"
        app_label = "django_llm"
        verbose_name = """ChainStep"""
        verbose_name_plural = """ChainSteps"""
        abstract = False

    def to_pydantic(self, context: "DjangoChainStepContext") -> ChainStep:
        """
        Convert this Django model to a Pydantic object.

        Args:
            context: Context instance containing values for non-serializable fields.
                    Required for this model because it has non-serializable fields.

        Returns:
            The corresponding ChainStep object

        Raises:
            ValueError: If context is missing required fields
        """
        return cast(ChainStep, super().to_pydantic(context=context))

"""
Django model for RetryStrategy.
"""

class DjangoRetryStrategy(Pydantic2DjangoBaseClass[RetryStrategy]):
    """
    Django model for RetryStrategy.
    """

    max_retries = models.IntegerField(verbose_name='max retries', default=3)
    delay = models.IntegerField(verbose_name='delay', default=1)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_retrystrategy"
        app_label = "django_llm"
        verbose_name = """RetryStrategy"""
        verbose_name_plural = """RetryStrategys"""
        abstract = False




# List of all generated models
__all__ = [
    'DjangoBasePrompt',
    'DjangoChainStep',
    'DjangoRetryStrategy',
    'DjangoChainStepContext',
]