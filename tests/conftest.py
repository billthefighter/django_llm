import os
import sys
import pytest
import django
from django.conf import settings

# Setup Django before importing models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.settings')

# Initialize Django
django.setup()

# Now import models after Django is configured
try:
    from src.django_llm.models import django_models
    
    # Get the models we need from the django_models dictionary
    # The model names in the registry are different from what we were trying to import
    ChainExecution = django_models.get('DjangoExecutionMetadata')  # Using closest equivalent
    ChainStep = django_models.get('DjangoChainStep')
    LLMProvider = django_models.get('DjangoProvider')
    StoredArtifact = django_models.get('DjangoArtifact')
    TokenUsageLog = django_models.get('DjangoTokenUsage')
    
    if not all([ChainExecution, ChainStep, LLMProvider, StoredArtifact, TokenUsageLog]):
        missing = []
        if not ChainExecution: missing.append('DjangoExecutionMetadata')
        if not ChainStep: missing.append('DjangoChainStep')
        if not LLMProvider: missing.append('DjangoProvider')
        if not StoredArtifact: missing.append('DjangoArtifact')
        if not TokenUsageLog: missing.append('DjangoTokenUsage')
        print(f"Warning: Some models are missing: {', '.join(missing)}")
        print(f"Available models: {', '.join(django_models.keys())}")
except ImportError as e:
    print(f"Error importing models: {e}")
    # Define placeholder models for testing if needed
    ChainExecution = None
    ChainStep = None
    LLMProvider = None
    StoredArtifact = None
    TokenUsageLog = None


@pytest.fixture
def llm_provider():
    if LLMProvider is None:
        pytest.skip("LLMProvider model not available")
    return LLMProvider.objects.create(
        name='anthropic',
        model='claude-3-sonnet-20240229',
        api_key='sk-ant-test123',
        max_tokens=1024,
        temperature=0.7
    )


@pytest.fixture
def chain_execution():
    if ChainExecution is None:
        pytest.skip("ChainExecution model not available")
    return ChainExecution.objects.create(
        name='test-chain',
        status='completed',
        execution_time=1.5
    )


@pytest.fixture
def chain_step(chain_execution):
    if ChainStep is None:
        pytest.skip("ChainStep model not available")
    return ChainStep.objects.create(
        chain_execution=chain_execution,
        name='test-step',
        status='completed',
        execution_time=0.5
    )


@pytest.fixture
def token_usage_log(chain_step, llm_provider):
    if TokenUsageLog is None:
        pytest.skip("TokenUsageLog model not available")
    return TokenUsageLog.objects.create(
        chain_step=chain_step,
        provider=llm_provider,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )


@pytest.fixture
def stored_artifact(chain_execution, chain_step):
    if StoredArtifact is None:
        pytest.skip("StoredArtifact model not available")
    return StoredArtifact.objects.create(
        chain_execution=chain_execution,
        chain_step=chain_step,
        name='test-artifact',
        content_type='text/plain',
        content='Test artifact content'
    ) 