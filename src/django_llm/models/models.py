"""
Generated Django models from Pydantic models.
Generated at: 2025-03-19 17:17:25
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
from typing import Callable, Dict, List, Optional

# Original Pydantic model imports
from llmaestro.agents.models import Agent, AgentMetrics, AgentResponse
from llmaestro.chains.chains import ChainContext, ChainEdge, ChainGraph, ChainMetadata, ChainNode, ChainState, ChainStep, ConditionalNode, DynamicPromptNode, RetryStrategy, ValidationNode
from llmaestro.chains.conversation_chain import ConversationChain, ConversationChainNode
from llmaestro.chains.tool_call_chain_example import ToolCallChain
from llmaestro.core.attachments import FileAttachment, ImageAttachment
from llmaestro.core.conversations import ConversationContext, ConversationEdge, ConversationGraph, ConversationNode
from llmaestro.core.graph import BaseEdge, BaseGraph, BaseGraph
from llmaestro.core.models import BaseResponse, ContextMetrics, LLMResponse, TokenUsage
from llmaestro.core.orchestrator import ExecutionMetadata
from llmaestro.core.storage import Artifact
from llmaestro.llm.credentials import APIKey
from llmaestro.llm.models import LLMInstance, LLMProfile, LLMRuntimeConfig, LLMState, Provider
from llmaestro.llm.rate_limiter import TokenBucket
from llmaestro.prompts.base import PromptVariable, VersionedPrompt
from llmaestro.prompts.mixins import VersionMixin
from llmaestro.prompts.types import PromptMetadata, VersionInfo

# Context class field type imports
from llmaestro.config.base import RateLimitConfig
from llmaestro.llm.capabilities import LLMCapabilities, ProviderCapabilities
from llmaestro.llm.models import LLMMetadata
from llmaestro.prompts.base import BasePrompt

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
            field_name="prompt",
            field_type=BasePrompt,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
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
        prompt: BasePrompt,
        input_transform: Optional[Callable],
        output_transform: Optional[Callable]    ) -> "DjangoChainStepContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            prompt: Value for prompt field
            input_transform: Value for input_transform field
            output_transform: Value for output_transform field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("prompt", prompt)
        context.set_value("input_transform", input_transform)
        context.set_value("output_transform", output_transform)
        return context

@dataclass
class DjangoChainGraphContext(ModelContext):
    """
    Context class for DjangoChainGraph.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoChainGraph"
    pydantic_class: Type = ChainGraph
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="agent_pool",
            field_type=llmaestro.agents.agent_pool.AgentPool,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        agent_pool: Optional[llmaestro.agents.agent_pool.AgentPool]    ) -> "DjangoChainGraphContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            agent_pool: Value for agent_pool field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("agent_pool", agent_pool)
        return context

@dataclass
class DjangoLLMResponseContext(ModelContext):
    """
    Context class for DjangoLLMResponse.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoLLMResponse"
    pydantic_class: Type = LLMResponse
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="context_metrics",
            field_type=llmaestro.core.models.ContextMetrics,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        context_metrics: Optional[llmaestro.core.models.ContextMetrics]    ) -> "DjangoLLMResponseContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            context_metrics: Value for context_metrics field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("context_metrics", context_metrics)
        return context

@dataclass
class DjangoProviderContext(ModelContext):
    """
    Context class for DjangoProvider.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoProvider"
    pydantic_class: Type = Provider
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="capabilities",
            field_type=ProviderCapabilities,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="rate_limits",
            field_type=RateLimitConfig,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        capabilities: ProviderCapabilities,
        rate_limits: RateLimitConfig    ) -> "DjangoProviderContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            capabilities: Value for capabilities field
            rate_limits: Value for rate_limits field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("capabilities", capabilities)
        context.set_value("rate_limits", rate_limits)
        return context

@dataclass
class DjangoLLMRuntimeConfigContext(ModelContext):
    """
    Context class for DjangoLLMRuntimeConfig.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoLLMRuntimeConfig"
    pydantic_class: Type = LLMRuntimeConfig
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="rate_limit",
            field_type=llmaestro.config.base.RateLimitConfig,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        rate_limit: Optional[llmaestro.config.base.RateLimitConfig]    ) -> "DjangoLLMRuntimeConfigContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            rate_limit: Value for rate_limit field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("rate_limit", rate_limit)
        return context

@dataclass
class DjangoLLMProfileContext(ModelContext):
    """
    Context class for DjangoLLMProfile.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoLLMProfile"
    pydantic_class: Type = LLMProfile
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="capabilities",
            field_type=LLMCapabilities,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="metadata",
            field_type=LLMMetadata,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="vision_capabilities",
            field_type=llmaestro.llm.capabilities.VisionCapabilities,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        capabilities: LLMCapabilities,
        metadata: LLMMetadata,
        vision_capabilities: Optional[llmaestro.llm.capabilities.VisionCapabilities]    ) -> "DjangoLLMProfileContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            capabilities: Value for capabilities field
            metadata: Value for metadata field
            vision_capabilities: Value for vision_capabilities field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("capabilities", capabilities)
        context.set_value("metadata", metadata)
        context.set_value("vision_capabilities", vision_capabilities)
        return context

@dataclass
class DjangoTokenBucketContext(ModelContext):
    """
    Context class for DjangoTokenBucket.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoTokenBucket"
    pydantic_class: Type = TokenBucket
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="rate_limit_config",
            field_type=RateLimitConfig,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        rate_limit_config: RateLimitConfig    ) -> "DjangoTokenBucketContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            rate_limit_config: Value for rate_limit_config field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("rate_limit_config", rate_limit_config)
        return context

@dataclass
class DjangoAgentContext(ModelContext):
    """
    Context class for DjangoAgent.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoAgent"
    pydantic_class: Type = Agent
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="capabilities",
            field_type=LLMCapabilities,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="metrics",
            field_type=llmaestro.agents.models.AgentMetrics,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        capabilities: LLMCapabilities,
        metrics: Optional[llmaestro.agents.models.AgentMetrics]    ) -> "DjangoAgentContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            capabilities: Value for capabilities field
            metrics: Value for metrics field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("capabilities", capabilities)
        context.set_value("metrics", metrics)
        return context

@dataclass
class DjangoVersionedPromptContext(ModelContext):
    """
    Context class for DjangoVersionedPrompt.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoVersionedPrompt"
    pydantic_class: Type = VersionedPrompt
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="attachments",
            field_type=List,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="expected_response",
            field_type=llmaestro.llm.responses.ResponseFormat,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="tools",
            field_type=List,
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        attachments: List,
        expected_response: Optional[llmaestro.llm.responses.ResponseFormat],
        tools: List    ) -> "DjangoVersionedPromptContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            attachments: Value for attachments field
            expected_response: Value for expected_response field
            tools: Value for tools field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("attachments", attachments)
        context.set_value("expected_response", expected_response)
        context.set_value("tools", tools)
        return context

@dataclass
class DjangoConversationChainContext(ModelContext):
    """
    Context class for DjangoConversationChain.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoConversationChain"
    pydantic_class: Type = ConversationChain
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="agent_pool",
            field_type=llmaestro.agents.agent_pool.AgentPool,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        agent_pool: Optional[llmaestro.agents.agent_pool.AgentPool]    ) -> "DjangoConversationChainContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            agent_pool: Value for agent_pool field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("agent_pool", agent_pool)
        return context

@dataclass
class DjangoConversationChainNodeContext(ModelContext):
    """
    Context class for DjangoConversationChainNode.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoConversationChainNode"
    pydantic_class: Type = ConversationChainNode
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="prompt",
            field_type=llmaestro.prompts.base.BasePrompt,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        prompt: Optional[llmaestro.prompts.base.BasePrompt]    ) -> "DjangoConversationChainNodeContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            prompt: Value for prompt field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("prompt", prompt)
        return context

@dataclass
class DjangoToolCallChainContext(ModelContext):
    """
    Context class for DjangoToolCallChain.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "DjangoToolCallChain"
    pydantic_class: Type = ToolCallChain
    django_model: Type[models.Model]
    context_fields: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="agent_pool",
            field_type=llmaestro.agents.agent_pool.AgentPool,
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls,
        django_model: Type[models.Model],
        agent_pool: Optional[llmaestro.agents.agent_pool.AgentPool]    ) -> "DjangoToolCallChainContext":
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            agent_pool: Value for agent_pool field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("agent_pool", agent_pool)
        return context


# Generated Django models
"""
Django model for RetryStrategy.
"""

class DjangoRetryStrategy(Pydantic2DjangoBaseClass[RetryStrategy]):
    """
    Django model for RetryStrategy.
    """

    max_retries = models.IntegerField(verbose_name='max retries', default=3)
    delay = models.FloatField(verbose_name='delay', default=1.0)
    exponential_backoff = models.BooleanField(verbose_name='exponential backoff', default=False)
    max_delay = models.FloatField(verbose_name='max delay', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_retrystrategy"
        app_label = "django_llm"
        verbose_name = """RetryStrategy"""
        verbose_name_plural = """RetryStrategys"""
        abstract = False


"""
Django model for ChainStep.
"""

class DjangoChainStep(Pydantic2DjangoBaseClass[ChainStep]):
    """
    Django model for ChainStep.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - prompt: BasePrompt
        - input_transform: Optional[Any]
        - output_transform: Optional[LLMResponse]
    """

    retry_strategy = models.ForeignKey(verbose_name='retry strategy', to='django_llm.RetryStrategy', on_delete=models.CASCADE)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chainstep"
        app_label = "django_llm"
        verbose_name = """ChainStep"""
        verbose_name_plural = """ChainSteps"""
        abstract = False

    def to_pydantic(self, context: "DjangoChainStepContext") -> ChainStep:
        """
        Convert this Django model to The corresponding ChainStep object.
        """
        return cast(ChainStep, super().to_pydantic(context=context))

"""
Django model for ChainMetadata.
"""

class DjangoChainMetadata(Pydantic2DjangoBaseClass[ChainMetadata]):
    """
    Django model for ChainMetadata.
    """

    description = models.TextField(verbose_name='description', null=True, blank=True)
    tags = models.JSONField(verbose_name='tags')
    version = models.TextField(verbose_name='version', null=True, blank=True)
    custom_data = models.JSONField(verbose_name='custom data')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chainmetadata"
        app_label = "django_llm"
        verbose_name = """ChainMetadata"""
        verbose_name_plural = """ChainMetadatas"""
        abstract = False


"""
Django model for ChainNode.
"""

class DjangoChainNode(Pydantic2DjangoBaseClass[ChainNode]):
    """
    Django model for ChainNode.
    """

    metadata = models.ForeignKey(verbose_name='metadata', to='django_llm.ChainMetadata', on_delete=models.CASCADE)
    step = models.ForeignKey(verbose_name='step', to='django_llm.ChainStep', on_delete=models.CASCADE)
    node_type = models.CharField(verbose_name='node type', choices=[('sequential', 'SEQUENTIAL'), ('parallel', 'PARALLEL'), ('conditional', 'CONDITIONAL'), ('agent', 'AGENT'), ('validation', 'VALIDATION')], max_length=11)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chainnode"
        app_label = "django_llm"
        verbose_name = """ChainNode"""
        verbose_name_plural = """ChainNodes"""
        abstract = False


"""
Django model for ChainEdge.
"""

class DjangoChainEdge(Pydantic2DjangoBaseClass[ChainEdge]):
    """
    Django model for ChainEdge.
    """

    source_id = models.TextField(verbose_name='source id', help_text='ID of the source node')
    target_id = models.TextField(verbose_name='target id', help_text='ID of the target node')
    edge_type = models.TextField(verbose_name='edge type', help_text='Type of relationship')
    metadata = models.JSONField(verbose_name='metadata')
    condition = models.TextField(verbose_name='condition', help_text='Optional condition for edge traversal', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chainedge"
        app_label = "django_llm"
        verbose_name = """ChainEdge"""
        verbose_name_plural = """ChainEdges"""
        abstract = False


"""
Django model for ChainState.
"""

class DjangoChainState(Pydantic2DjangoBaseClass[ChainState]):
    """
    Django model for ChainState.
    """

    status = models.TextField(verbose_name='status', default='pending')
    current_step = models.TextField(verbose_name='current step', null=True, blank=True)
    completed_steps = models.JSONField(verbose_name='completed steps')
    failed_steps = models.JSONField(verbose_name='failed steps')
    step_results = models.JSONField(verbose_name='step results')
    variables = models.JSONField(verbose_name='variables')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chainstate"
        app_label = "django_llm"
        verbose_name = """ChainState"""
        verbose_name_plural = """ChainStates"""
        abstract = False


"""
Django model for ChainContext.
"""

class DjangoChainContext(Pydantic2DjangoBaseClass[ChainContext]):
    """
    Django model for ChainContext.
    """

    metadata = models.ForeignKey(verbose_name='metadata', to='django_llm.ChainMetadata', on_delete=models.CASCADE)
    state = models.ForeignKey(verbose_name='state', to='django_llm.ChainState', on_delete=models.CASCADE)
    variables = models.JSONField(verbose_name='variables')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chaincontext"
        app_label = "django_llm"
        verbose_name = """ChainContext"""
        verbose_name_plural = """ChainContexts"""
        abstract = False


"""
Django model for ChainGraph.
"""

class DjangoChainGraph(Pydantic2DjangoBaseClass[ChainGraph]):
    """
    Django model for ChainGraph.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - agent_pool: Optional[AgentPool]
    """

    metadata = models.JSONField(verbose_name='metadata')
    context = models.ForeignKey(verbose_name='context', to='django_llm.ChainContext', on_delete=models.CASCADE)
    verify_acyclic = models.BooleanField(verbose_name='verify acyclic', help_text='Whether to verify the graph is acyclic during initialization', default=True)
    nodes = models.ManyToManyField(verbose_name='nodes', to='django_llm.ChainNode')
    edges = models.ManyToManyField(verbose_name='edges', to='django_llm.ChainEdge')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_chaingraph"
        app_label = "django_llm"
        verbose_name = """ChainGraph"""
        verbose_name_plural = """ChainGraphs"""
        abstract = False

    def to_pydantic(self, context: "DjangoChainGraphContext") -> ChainGraph:
        """
        Convert this Django model to The corresponding ChainGraph object.
        """
        return cast(ChainGraph, super().to_pydantic(context=context))

"""
Django model for ConversationEdge.
"""

class DjangoConversationEdge(Pydantic2DjangoBaseClass[ConversationEdge]):
    """
    Django model for ConversationEdge.
    """

    source_id = models.TextField(verbose_name='source id', help_text='ID of the source node')
    target_id = models.TextField(verbose_name='target id', help_text='ID of the target node')
    edge_type = models.TextField(verbose_name='edge type', help_text='Type of relationship')
    metadata = models.JSONField(verbose_name='metadata')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_conversationedge"
        app_label = "django_llm"
        verbose_name = """ConversationEdge"""
        verbose_name_plural = """ConversationEdges"""
        abstract = False


"""
Django model for ConversationNode.
"""

class DjangoConversationNode(Pydantic2DjangoBaseClass[ConversationNode]):
    """
    Django model for ConversationNode.
    """

    metadata = models.JSONField(verbose_name='metadata')
    content = models.JSONField(verbose_name='content', help_text='The prompt or response content')
    node_type = models.TextField(verbose_name='node type', help_text='Type of node (prompt/response)')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_conversationnode"
        app_label = "django_llm"
        verbose_name = """ConversationNode"""
        verbose_name_plural = """ConversationNodes"""
        abstract = False


"""
Django model for ConversationGraph.
"""

class DjangoConversationGraph(Pydantic2DjangoBaseClass[ConversationGraph]):
    """
    Django model for ConversationGraph.
    """

    metadata = models.JSONField(verbose_name='metadata')
    nodes = models.ManyToManyField(verbose_name='nodes', to='django_llm.ConversationNode')
    edges = models.ManyToManyField(verbose_name='edges', to='django_llm.ConversationEdge')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_conversationgraph"
        app_label = "django_llm"
        verbose_name = """ConversationGraph"""
        verbose_name_plural = """ConversationGraphs"""
        abstract = False


"""
Django model for BaseEdge.
"""

class DjangoBaseEdge(Pydantic2DjangoBaseClass[BaseEdge]):
    """
    Django model for BaseEdge.
    """

    source_id = models.TextField(verbose_name='source id', help_text='ID of the source node')
    target_id = models.TextField(verbose_name='target id', help_text='ID of the target node')
    edge_type = models.TextField(verbose_name='edge type', help_text='Type of relationship')
    metadata = models.JSONField(verbose_name='metadata')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_baseedge"
        app_label = "django_llm"
        verbose_name = """BaseEdge"""
        verbose_name_plural = """BaseEdges"""
        abstract = False


"""
Django model for BaseGraph.
"""

class DjangoBaseGraph(Pydantic2DjangoBaseClass[BaseGraph]):
    """
    Django model for BaseGraph.
    """

    nodes = models.JSONField(verbose_name='nodes')
    edges = models.JSONField(verbose_name='edges')
    metadata = models.JSONField(verbose_name='metadata')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_basegraph"
        app_label = "django_llm"
        verbose_name = """BaseGraph"""
        verbose_name_plural = """BaseGraphs"""
        abstract = False


"""
Django model for BaseNode.
"""

class DjangoBaseNode(Pydantic2DjangoBaseClass[BaseNode]):
    """
    Django model for BaseNode.
    """

    metadata = models.JSONField(verbose_name='metadata')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_basenode"
        app_label = "django_llm"
        verbose_name = """BaseNode"""
        verbose_name_plural = """BaseNodes"""
        abstract = False


"""
Django model for TokenUsage.
"""

class DjangoTokenUsage(Pydantic2DjangoBaseClass[TokenUsage]):
    """
    Django model for TokenUsage.
    """

    prompt_tokens = models.IntegerField(verbose_name='prompt tokens')
    completion_tokens = models.IntegerField(verbose_name='completion tokens')
    total_tokens = models.IntegerField(verbose_name='total tokens')
    estimated_cost = models.FloatField(verbose_name='estimated cost', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_tokenusage"
        app_label = "django_llm"
        verbose_name = """TokenUsage"""
        verbose_name_plural = """TokenUsages"""
        abstract = False


"""
Django model for LLMResponse.
"""

class DjangoLLMResponse(Pydantic2DjangoBaseClass[LLMResponse]):
    """
    Django model for LLMResponse.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - context_metrics: Optional[ContextMetrics]
    """

    timestamp = models.DateTimeField(verbose_name='timestamp')
    success = models.BooleanField(verbose_name='success', help_text='Whether the operation was successful')
    error = models.TextField(verbose_name='error', help_text='Error message if operation failed', null=True, blank=True)
    execution_time = models.FloatField(verbose_name='execution time', help_text='Time taken to generate response in seconds', null=True, blank=True)
    metadata = models.JSONField(verbose_name='metadata', help_text='Additional metadata about the response')
    content = models.TextField(verbose_name='content', help_text='The content of the response')
    token_usage = models.ForeignKey(verbose_name='token usage', help_text='Token usage statistics', to='django_llm.TokenUsage', on_delete=models.CASCADE)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_llmresponse"
        app_label = "django_llm"
        verbose_name = """LLMResponse"""
        verbose_name_plural = """LLMResponses"""
        abstract = False

    def to_pydantic(self, context: "DjangoLLMResponseContext") -> LLMResponse:
        """
        Convert this Django model to The corresponding LLMResponse object.
        """
        return cast(LLMResponse, super().to_pydantic(context=context))

"""
Django model for BaseResponse.
"""

class DjangoBaseResponse(Pydantic2DjangoBaseClass[BaseResponse]):
    """
    Django model for BaseResponse.
    """

    timestamp = models.DateTimeField(verbose_name='timestamp')
    success = models.BooleanField(verbose_name='success', help_text='Whether the operation was successful')
    error = models.TextField(verbose_name='error', help_text='Error message if operation failed', null=True, blank=True)
    execution_time = models.FloatField(verbose_name='execution time', help_text='Time taken to generate response in seconds', null=True, blank=True)
    metadata = models.JSONField(verbose_name='metadata', help_text='Additional metadata about the response')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_baseresponse"
        app_label = "django_llm"
        verbose_name = """BaseResponse"""
        verbose_name_plural = """BaseResponses"""
        abstract = False


"""
Django model for ContextMetrics.
"""

class DjangoContextMetrics(Pydantic2DjangoBaseClass[ContextMetrics]):
    """
    Django model for ContextMetrics.
    """

    max_context_tokens = models.IntegerField(verbose_name='max context tokens')
    current_context_tokens = models.IntegerField(verbose_name='current context tokens')
    available_tokens = models.IntegerField(verbose_name='available tokens')
    context_utilization = models.FloatField(verbose_name='context utilization')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_contextmetrics"
        app_label = "django_llm"
        verbose_name = """ContextMetrics"""
        verbose_name_plural = """ContextMetricss"""
        abstract = False


"""
Django model for APIKey.
"""

class DjangoAPIKey(Pydantic2DjangoBaseClass[APIKey]):
    """
    Django model for APIKey.
    """

    key = models.TextField(verbose_name='key', help_text='The actual API key value')
    is_encrypted = models.BooleanField(verbose_name='is encrypted', help_text='Whether the key is encrypted', default=False)
    last_used = models.DateTimeField(verbose_name='last used', help_text='When this key was last used', null=True, blank=True)
    last_rotated = models.DateTimeField(verbose_name='last rotated', help_text='When this key was last rotated', null=True, blank=True)
    expiration = models.DateTimeField(verbose_name='expiration', help_text='When this key expires', null=True, blank=True)
    description = models.TextField(verbose_name='description', help_text='Optional description or notes', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_apikey"
        app_label = "django_llm"
        verbose_name = """APIKey"""
        verbose_name_plural = """APIKeys"""
        abstract = False


"""
Django model for Provider.
"""

class DjangoProvider(Pydantic2DjangoBaseClass[Provider]):
    """
    Django model for Provider.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - capabilities: ProviderCapabilities
        - rate_limits: RateLimitConfig
    """

    family = models.TextField(verbose_name='family')
    description = models.TextField(verbose_name='description', null=True, blank=True)
    api_base = models.TextField(verbose_name='api base', help_text='Base URL for the provider\'s API')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_provider"
        app_label = "django_llm"
        verbose_name = """Provider"""
        verbose_name_plural = """Providers"""
        abstract = False

    def to_pydantic(self, context: "DjangoProviderContext") -> Provider:
        """
        Convert this Django model to The corresponding Provider object.
        """
        return cast(Provider, super().to_pydantic(context=context))

"""
Django model for LLMRuntimeConfig.
"""

class DjangoLLMRuntimeConfig(Pydantic2DjangoBaseClass[LLMRuntimeConfig]):
    """
    Django model for LLMRuntimeConfig.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - rate_limit: Optional[RateLimitConfig]
    """

    max_tokens = models.IntegerField(verbose_name='max tokens', help_text='Maximum number of tokens to generate', default=2048)
    temperature = models.FloatField(verbose_name='temperature', help_text='Sampling temperature (None means use model default)', null=True, blank=True)
    max_context_tokens = models.IntegerField(verbose_name='max context tokens', help_text='Maximum context window size', default=4096)
    stream = models.BooleanField(verbose_name='stream', help_text='Whether to stream the response', default=False)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_llmruntimeconfig"
        app_label = "django_llm"
        verbose_name = """LLMRuntimeConfig"""
        verbose_name_plural = """LLMRuntimeConfigs"""
        abstract = False

    def to_pydantic(self, context: "DjangoLLMRuntimeConfigContext") -> LLMRuntimeConfig:
        """
        Convert this Django model to The corresponding LLMRuntimeConfig object.
        """
        return cast(LLMRuntimeConfig, super().to_pydantic(context=context))

"""
Django model for LLMProfile.
"""

class DjangoLLMProfile(Pydantic2DjangoBaseClass[LLMProfile]):
    """
    Django model for LLMProfile.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - capabilities: LLMCapabilities
        - metadata: LLMMetadata
        - vision_capabilities: Optional[VisionCapabilities]
    """

    version = models.TextField(verbose_name='version', null=True, blank=True)
    description = models.TextField(verbose_name='description', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_llmprofile"
        app_label = "django_llm"
        verbose_name = """LLMProfile"""
        verbose_name_plural = """LLMProfiles"""
        abstract = False

    def to_pydantic(self, context: "DjangoLLMProfileContext") -> LLMProfile:
        """
        Convert this Django model to The corresponding LLMProfile object.
        """
        return cast(LLMProfile, super().to_pydantic(context=context))

"""
Django model for LLMState.
"""

class DjangoLLMState(Pydantic2DjangoBaseClass[LLMState]):
    """
    Django model for LLMState.
    """

    profile = models.ForeignKey(verbose_name='profile', help_text='Model profile containing capabilities and metadata', to='django_llm.LLMProfile', on_delete=models.CASCADE)
    provider = models.ForeignKey(verbose_name='provider', help_text='Provider configuration', to='django_llm.Provider', on_delete=models.CASCADE)
    runtime_config = models.ForeignKey(verbose_name='runtime config', help_text='Runtime configuration', to='django_llm.LLMRuntimeConfig', on_delete=models.CASCADE)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_llmstate"
        app_label = "django_llm"
        verbose_name = """LLMState"""
        verbose_name_plural = """LLMStates"""
        abstract = False


"""
Django model for LLMInstance.
"""

class DjangoLLMInstance(Pydantic2DjangoBaseClass[LLMInstance]):
    """
    Django model for LLMInstance.
    """

    state = models.ForeignKey(verbose_name='state', help_text='Configuration and metadata for the LLM', to='django_llm.LLMState', on_delete=models.CASCADE)
    interface = models.JSONField(verbose_name='interface', help_text='Active interface instance for LLM interactions')
    credentials = models.ForeignKey(verbose_name='credentials', help_text='API credentials for this instance', null=True, blank=True, to='django_llm.APIKey', on_delete=models.CASCADE)
    is_initialized = models.BooleanField(verbose_name='is initialized', help_text='Whether this instance has been fully initialized', default=False)
    is_healthy = models.BooleanField(verbose_name='is healthy', help_text='Whether this instance is currently healthy and operational', default=True)
    last_error = models.TextField(verbose_name='last error', help_text='Last error encountered by this instance', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_llminstance"
        app_label = "django_llm"
        verbose_name = """LLMInstance"""
        verbose_name_plural = """LLMInstances"""
        abstract = False


"""
Django model for TokenBucket.
"""

class DjangoTokenBucket(Pydantic2DjangoBaseClass[TokenBucket]):
    """
    Django model for TokenBucket.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - rate_limit_config: RateLimitConfig
    """

    daily_usage = models.JSONField(verbose_name='daily usage', help_text='Daily token usage mapped by date')
    minute_tokens = models.IntegerField(verbose_name='minute tokens', help_text='Current number of tokens in the minute bucket', default=0)
    last_refill_timestamp = models.FloatField(verbose_name='last refill timestamp', help_text='Last time the buckets were refilled')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_tokenbucket"
        app_label = "django_llm"
        verbose_name = """TokenBucket"""
        verbose_name_plural = """TokenBuckets"""
        abstract = False

    def to_pydantic(self, context: "DjangoTokenBucketContext") -> TokenBucket:
        """
        Convert this Django model to The corresponding TokenBucket object.
        """
        return cast(TokenBucket, super().to_pydantic(context=context))

"""
Django model for FileAttachment.
"""

class DjangoFileAttachment(Pydantic2DjangoBaseClass[FileAttachment]):
    """
    Django model for FileAttachment.
    """

    content = models.JSONField(verbose_name='content', help_text='The file content, either as string or bytes')
    media_type = models.CharField(verbose_name='media type', help_text='The media type of the file', choices=[('image/jpeg', 'JPEG'), ('image/png', 'PNG'), ('image/gif', 'GIF'), ('image/webp', 'WEBP'), ('image/bmp', 'BMP'), ('image/tiff', 'TIFF'), ('image/svg+xml', 'SVG'), ('application/pdf', 'PDF'), ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'DOCX'), ('application/msword', 'DOC'), ('text/plain', 'TXT'), ('application/octet-stream', 'UNKNOWN')], max_length=71)
    file_name = models.TextField(verbose_name='file name', help_text='Name of the file', null=True, blank=True)
    description = models.TextField(verbose_name='description', help_text='Optional description of the file', null=True, blank=True)
    file_id = models.TextField(verbose_name='file id', help_text='ID of the file when uploaded to an LLM provider', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_fileattachment"
        app_label = "django_llm"
        verbose_name = """FileAttachment"""
        verbose_name_plural = """FileAttachments"""
        abstract = False


"""
Django model for ImageAttachment.
"""

class DjangoImageAttachment(Pydantic2DjangoBaseClass[ImageAttachment]):
    """
    Django model for ImageAttachment.
    """

    content = models.JSONField(verbose_name='content', help_text='The file content, either as string or bytes')
    media_type = models.CharField(verbose_name='media type', help_text='The media type of the file', choices=[('image/jpeg', 'JPEG'), ('image/png', 'PNG'), ('image/gif', 'GIF'), ('image/webp', 'WEBP'), ('image/bmp', 'BMP'), ('image/tiff', 'TIFF'), ('image/svg+xml', 'SVG'), ('application/pdf', 'PDF'), ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'DOCX'), ('application/msword', 'DOC'), ('text/plain', 'TXT'), ('application/octet-stream', 'UNKNOWN')], max_length=71)
    file_name = models.TextField(verbose_name='file name', help_text='Name of the file', null=True, blank=True)
    description = models.TextField(verbose_name='description', help_text='Optional description of the file', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_imageattachment"
        app_label = "django_llm"
        verbose_name = """ImageAttachment"""
        verbose_name_plural = """ImageAttachments"""
        abstract = False


"""
Django model for ExecutionMetadata.
"""

class DjangoExecutionMetadata(Pydantic2DjangoBaseClass[ExecutionMetadata]):
    """
    Django model for ExecutionMetadata.
    """

    status = models.TextField(verbose_name='status', default='pending')
    started_at = models.DateTimeField(verbose_name='started at', null=True, blank=True)
    completed_at = models.DateTimeField(verbose_name='completed at', null=True, blank=True)
    error = models.TextField(verbose_name='error', null=True, blank=True)
    parallel_group = models.TextField(verbose_name='parallel group', null=True, blank=True)
    dependencies = models.JSONField(verbose_name='dependencies', default=[])

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_executionmetadata"
        app_label = "django_llm"
        verbose_name = """ExecutionMetadata"""
        verbose_name_plural = """ExecutionMetadatas"""
        abstract = False


"""
Django model for Artifact.
"""

class DjangoArtifact(Pydantic2DjangoBaseClass[Artifact]):
    """
    Django model for Artifact.
    """

    content_type = models.TextField(verbose_name='content type')
    data = models.JSONField(verbose_name='data')
    path = models.FilePathField(verbose_name='path', null=True, blank=True)
    timestamp = models.DateTimeField(verbose_name='timestamp')
    metadata = models.JSONField(verbose_name='metadata')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_artifact"
        app_label = "django_llm"
        verbose_name = """Artifact"""
        verbose_name_plural = """Artifacts"""
        abstract = False


"""
Django model for ConversationContext.
"""

class DjangoConversationContext(Pydantic2DjangoBaseClass[ConversationContext]):
    """
    Django model for ConversationContext.
    """

    graph = models.ForeignKey(verbose_name='graph', help_text='The underlying conversation graph', to='django_llm.ConversationGraph', on_delete=models.CASCADE)
    current_message = models.TextField(verbose_name='current message', help_text='ID of the current node being processed', null=True, blank=True)
    max_nodes = models.IntegerField(verbose_name='max nodes', help_text='Maximum number of nodes before auto-pruning. None means no auto-pruning.', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_conversationcontext"
        app_label = "django_llm"
        verbose_name = """ConversationContext"""
        verbose_name_plural = """ConversationContexts"""
        abstract = False


"""
Django model for Agent.
"""

class DjangoAgent(Pydantic2DjangoBaseClass[Agent]):
    """
    Django model for Agent.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - capabilities: LLMCapabilities
        - metrics: Optional[AgentMetrics]
    """

    type = models.TextField(verbose_name='type', help_text='Type of agent (e.g., \'general\', \'specialist\')')
    model = models.TextField(verbose_name='model', help_text='The model being used by this agent')
    state = models.CharField(verbose_name='state', choices=[('idle', 'IDLE'), ('busy', 'BUSY'), ('error', 'ERROR'), ('terminated', 'TERMINATED')], default='AgentState.IDLE', max_length=10)
    metadata = models.JSONField(verbose_name='metadata')
    last_active = models.DateTimeField(verbose_name='last active')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_agent"
        app_label = "django_llm"
        verbose_name = """Agent"""
        verbose_name_plural = """Agents"""
        abstract = False

    def to_pydantic(self, context: "DjangoAgentContext") -> Agent:
        """
        Convert this Django model to The corresponding Agent object.
        """
        return cast(Agent, super().to_pydantic(context=context))

"""
Django model for AgentMetrics.
"""

class DjangoAgentMetrics(Pydantic2DjangoBaseClass[AgentMetrics]):
    """
    Django model for AgentMetrics.
    """

    token_usage = models.ForeignKey(verbose_name='token usage', to='django_llm.TokenUsage', on_delete=models.CASCADE)
    context_metrics = models.ForeignKey(verbose_name='context metrics', to='django_llm.ContextMetrics', on_delete=models.CASCADE)
    execution_time = models.FloatField(verbose_name='execution time')
    timestamp = models.DateTimeField(verbose_name='timestamp')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_agentmetrics"
        app_label = "django_llm"
        verbose_name = """AgentMetrics"""
        verbose_name_plural = """AgentMetricss"""
        abstract = False


"""
Django model for AgentResponse.
"""

class DjangoAgentResponse(Pydantic2DjangoBaseClass[AgentResponse]):
    """
    Django model for AgentResponse.
    """

    timestamp = models.DateTimeField(verbose_name='timestamp')
    success = models.BooleanField(verbose_name='success', help_text='Whether the operation was successful')
    error = models.TextField(verbose_name='error', help_text='Error message if operation failed', null=True, blank=True)
    execution_time = models.FloatField(verbose_name='execution time', help_text='Time taken to generate response in seconds', null=True, blank=True)
    metadata = models.JSONField(verbose_name='metadata', help_text='Additional metadata about the response')
    agent_id = models.TextField(verbose_name='agent id', help_text='ID of the agent that generated the response')
    agent_type = models.TextField(verbose_name='agent type', help_text='Type of agent that generated the response')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_agentresponse"
        app_label = "django_llm"
        verbose_name = """AgentResponse"""
        verbose_name_plural = """AgentResponses"""
        abstract = False


"""
Django model for VersionInfo.
"""

class DjangoVersionInfo(Pydantic2DjangoBaseClass[VersionInfo]):
    """
    Django model for VersionInfo.
    """

    number = models.TextField(verbose_name='number')
    timestamp = models.DateTimeField(verbose_name='timestamp')
    author = models.TextField(verbose_name='author')
    description = models.TextField(verbose_name='description')
    change_type = models.TextField(verbose_name='change type')
    git_commit = models.TextField(verbose_name='git commit', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_versioninfo"
        app_label = "django_llm"
        verbose_name = """VersionInfo"""
        verbose_name_plural = """VersionInfos"""
        abstract = False


"""
Django model for VersionMixin.
"""

class DjangoVersionMixin(Pydantic2DjangoBaseClass[VersionMixin]):
    """
    Django model for VersionMixin.
    """

    current_version = models.ForeignKey(verbose_name='current version', to='django_llm.VersionInfo', on_delete=models.CASCADE)
    version_history = models.ManyToManyField(verbose_name='version history', to='django_llm.VersionInfo')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_versionmixin"
        app_label = "django_llm"
        verbose_name = """VersionMixin"""
        verbose_name_plural = """VersionMixins"""
        abstract = False


"""
Django model for PromptMetadata.
"""

class DjangoPromptMetadata(Pydantic2DjangoBaseClass[PromptMetadata]):
    """
    Django model for PromptMetadata.
    """

    type = models.TextField(verbose_name='type')
    model_requirements = models.JSONField(verbose_name='model requirements', null=True, blank=True)
    decomposition = models.JSONField(verbose_name='decomposition', null=True, blank=True)
    tags = models.JSONField(verbose_name='tags')
    is_active = models.BooleanField(verbose_name='is active', default=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_promptmetadata"
        app_label = "django_llm"
        verbose_name = """PromptMetadata"""
        verbose_name_plural = """PromptMetadatas"""
        abstract = False


"""
Django model for PromptVariable.
"""

class DjangoPromptVariable(Pydantic2DjangoBaseClass[PromptVariable]):
    """
    Django model for PromptVariable.
    """

    description = models.TextField(verbose_name='description', null=True, blank=True)
    expected_input_type = models.CharField(verbose_name='expected input type', choices=[('string', 'STRING'), ('integer', 'INTEGER'), ('float', 'FLOAT'), ('boolean', 'BOOLEAN'), ('list', 'LIST'), ('dict', 'DICT'), ('schema', 'SCHEMA')], default='SerializableType.STRING', max_length=7)
    string_conversion_template = models.TextField(verbose_name='string conversion template')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_promptvariable"
        app_label = "django_llm"
        verbose_name = """PromptVariable"""
        verbose_name_plural = """PromptVariables"""
        abstract = False


"""
Django model for VersionedPrompt.
"""

class DjangoVersionedPrompt(Pydantic2DjangoBaseClass[VersionedPrompt]):
    """
    Django model for VersionedPrompt.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - attachments: List[BaseAttachment]
        - expected_response: Optional[ResponseFormat]
        - tools: List[ToolParams]
    """

    current_version = models.ForeignKey(verbose_name='current version', to='django_llm.VersionInfo', on_delete=models.CASCADE)
    description = models.TextField(verbose_name='description')
    system_prompt = models.TextField(verbose_name='system prompt', help_text='System prompt template')
    user_prompt = models.TextField(verbose_name='user prompt', help_text='User prompt template')
    metadata = models.ForeignKey(verbose_name='metadata', help_text='Optional metadata about the prompt', null=True, blank=True, to='django_llm.PromptMetadata', on_delete=models.CASCADE)
    examples = models.JSONField(verbose_name='examples', null=True, blank=True)
    template_schema = models.JSONField(verbose_name='template schema', help_text='Optional schema for template validation', null=True, blank=True)
    version_history = models.ManyToManyField(verbose_name='version history', to='django_llm.VersionInfo')
    variables = models.ManyToManyField(verbose_name='variables', help_text='List of variable definitions', to='django_llm.PromptVariable')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_versionedprompt"
        app_label = "django_llm"
        verbose_name = """VersionedPrompt"""
        verbose_name_plural = """VersionedPrompts"""
        abstract = False

    def to_pydantic(self, context: "DjangoVersionedPromptContext") -> VersionedPrompt:
        """
        Convert this Django model to The corresponding VersionedPrompt object.
        """
        return cast(VersionedPrompt, super().to_pydantic(context=context))

"""
Django model for ConditionalNode.
"""

class DjangoConditionalNode(Pydantic2DjangoBaseClass[ConditionalNode]):
    """
    Django model for ConditionalNode.
    """

    metadata = models.ForeignKey(verbose_name='metadata', to='django_llm.ChainMetadata', on_delete=models.CASCADE)
    step = models.ForeignKey(verbose_name='step', to='django_llm.ChainStep', on_delete=models.CASCADE)
    node_type = models.CharField(verbose_name='node type', choices=[('sequential', 'SEQUENTIAL'), ('parallel', 'PARALLEL'), ('conditional', 'CONDITIONAL'), ('agent', 'AGENT'), ('validation', 'VALIDATION')], max_length=11)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_conditionalnode"
        app_label = "django_llm"
        verbose_name = """ConditionalNode"""
        verbose_name_plural = """ConditionalNodes"""
        abstract = False


"""
Django model for DynamicPromptNode.
"""

class DjangoDynamicPromptNode(Pydantic2DjangoBaseClass[DynamicPromptNode]):
    """
    Django model for DynamicPromptNode.
    """

    metadata = models.ForeignKey(verbose_name='metadata', to='django_llm.ChainMetadata', on_delete=models.CASCADE)
    step = models.ForeignKey(verbose_name='step', to='django_llm.ChainStep', on_delete=models.CASCADE)
    node_type = models.CharField(verbose_name='node type', choices=[('sequential', 'SEQUENTIAL'), ('parallel', 'PARALLEL'), ('conditional', 'CONDITIONAL'), ('agent', 'AGENT'), ('validation', 'VALIDATION')], max_length=11)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_dynamicpromptnode"
        app_label = "django_llm"
        verbose_name = """DynamicPromptNode"""
        verbose_name_plural = """DynamicPromptNodes"""
        abstract = False


"""
Django model for ValidationNode.
"""

class DjangoValidationNode(Pydantic2DjangoBaseClass[ValidationNode]):
    """
    Django model for ValidationNode.
    """

    metadata = models.ForeignKey(verbose_name='metadata', to='django_llm.ChainMetadata', on_delete=models.CASCADE)
    step = models.ForeignKey(verbose_name='step', to='django_llm.ChainStep', on_delete=models.CASCADE)
    node_type = models.CharField(verbose_name='node type', choices=[('sequential', 'SEQUENTIAL'), ('parallel', 'PARALLEL'), ('conditional', 'CONDITIONAL'), ('agent', 'AGENT'), ('validation', 'VALIDATION')], max_length=11)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_validationnode"
        app_label = "django_llm"
        verbose_name = """ValidationNode"""
        verbose_name_plural = """ValidationNodes"""
        abstract = False


"""
Django model for ConversationChain.
"""

class DjangoConversationChain(Pydantic2DjangoBaseClass[ConversationChain]):
    """
    Django model for ConversationChain.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - agent_pool: Optional[AgentPool]
    """

    metadata = models.JSONField(verbose_name='metadata')
    context = models.ForeignKey(verbose_name='context', to='django_llm.ChainContext', on_delete=models.CASCADE)
    verify_acyclic = models.BooleanField(verbose_name='verify acyclic', help_text='Whether to verify the graph is acyclic during initialization', default=True)
    nodes = models.ManyToManyField(verbose_name='nodes', to='django_llm.ChainNode')
    edges = models.ManyToManyField(verbose_name='edges', to='django_llm.ChainEdge')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_conversationchain"
        app_label = "django_llm"
        verbose_name = """ConversationChain"""
        verbose_name_plural = """ConversationChains"""
        abstract = False

    def to_pydantic(self, context: "DjangoConversationChainContext") -> ConversationChain:
        """
        Convert this Django model to The corresponding ConversationChain object.
        """
        return cast(ConversationChain, super().to_pydantic(context=context))

"""
Django model for ConversationChainNode.
"""

class DjangoConversationChainNode(Pydantic2DjangoBaseClass[ConversationChainNode]):
    """
    Django model for ConversationChainNode.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - prompt: Optional[BasePrompt]
    """

    metadata = models.ForeignKey(verbose_name='metadata', to='django_llm.ChainMetadata', on_delete=models.CASCADE)
    step = models.ForeignKey(verbose_name='step', to='django_llm.ChainStep', on_delete=models.CASCADE)
    node_type = models.CharField(verbose_name='node type', choices=[('sequential', 'SEQUENTIAL'), ('parallel', 'PARALLEL'), ('conditional', 'CONDITIONAL'), ('agent', 'AGENT'), ('validation', 'VALIDATION')], max_length=11)
    conversation_id = models.TextField(verbose_name='conversation id', null=True, blank=True)
    prompt_node_id = models.TextField(verbose_name='prompt node id', null=True, blank=True)
    response_node_id = models.TextField(verbose_name='response node id', null=True, blank=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_conversationchainnode"
        app_label = "django_llm"
        verbose_name = """ConversationChainNode"""
        verbose_name_plural = """ConversationChainNodes"""
        abstract = False

    def to_pydantic(self, context: "DjangoConversationChainNodeContext") -> ConversationChainNode:
        """
        Convert this Django model to The corresponding ConversationChainNode object.
        """
        return cast(ConversationChainNode, super().to_pydantic(context=context))

"""
Django model for ToolCallChain.
"""

class DjangoToolCallChain(Pydantic2DjangoBaseClass[ToolCallChain]):
    """
    Django model for ToolCallChain.

    Context Fields:
        The following fields require context when converting back to Pydantic:
        - agent_pool: Optional[AgentPool]
    """

    metadata = models.JSONField(verbose_name='metadata')
    context = models.ForeignKey(verbose_name='context', to='django_llm.ChainContext', on_delete=models.CASCADE)
    verify_acyclic = models.BooleanField(verbose_name='verify acyclic', help_text='Whether to verify the graph is acyclic during initialization', default=True)
    nodes = models.ManyToManyField(verbose_name='nodes', to='django_llm.ChainNode')
    edges = models.ManyToManyField(verbose_name='edges', to='django_llm.ChainEdge')

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "django_llm_toolcallchain"
        app_label = "django_llm"
        verbose_name = """ToolCallChain"""
        verbose_name_plural = """ToolCallChains"""
        abstract = False

    def to_pydantic(self, context: "DjangoToolCallChainContext") -> ToolCallChain:
        """
        Convert this Django model to The corresponding ToolCallChain object.
        """
        return cast(ToolCallChain, super().to_pydantic(context=context))



# List of all generated models
__all__ = [
    'DjangoRetryStrategy',
    'DjangoChainStep',
    'DjangoChainMetadata',
    'DjangoChainNode',
    'DjangoChainEdge',
    'DjangoChainState',
    'DjangoChainContext',
    'DjangoChainGraph',
    'DjangoConversationEdge',
    'DjangoConversationNode',
    'DjangoConversationGraph',
    'DjangoBaseEdge',
    'DjangoBaseGraph',
    'DjangoBaseNode',
    'DjangoTokenUsage',
    'DjangoLLMResponse',
    'DjangoBaseResponse',
    'DjangoContextMetrics',
    'DjangoAPIKey',
    'DjangoProvider',
    'DjangoLLMRuntimeConfig',
    'DjangoLLMProfile',
    'DjangoLLMState',
    'DjangoLLMInstance',
    'DjangoTokenBucket',
    'DjangoBaseGraph',
    'DjangoBaseGraph',
    'DjangoFileAttachment',
    'DjangoImageAttachment',
    'DjangoExecutionMetadata',
    'DjangoArtifact',
    'DjangoConversationContext',
    'DjangoAgent',
    'DjangoAgentMetrics',
    'DjangoAgentResponse',
    'DjangoVersionInfo',
    'DjangoVersionMixin',
    'DjangoPromptMetadata',
    'DjangoPromptVariable',
    'DjangoVersionedPrompt',
    'DjangoConditionalNode',
    'DjangoDynamicPromptNode',
    'DjangoValidationNode',
    'DjangoConversationChain',
    'DjangoConversationChainNode',
    'DjangoToolCallChain',
    'DjangoChainStepContext',
    'DjangoChainGraphContext',
    'DjangoLLMResponseContext',
    'DjangoProviderContext',
    'DjangoLLMRuntimeConfigContext',
    'DjangoLLMProfileContext',
    'DjangoTokenBucketContext',
    'DjangoAgentContext',
    'DjangoVersionedPromptContext',
    'DjangoConversationChainContext',
    'DjangoConversationChainNodeContext',
    'DjangoToolCallChainContext',
]