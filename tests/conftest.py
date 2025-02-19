import pytest

from src.django_llm.models import (
    ChainExecution,
    ChainStep,
    LLMProvider,
    StoredArtifact,
    TokenUsageLog,
)


@pytest.fixture
def llm_provider():
    return LLMProvider.objects.create(
        name='anthropic',
        model='claude-3-sonnet-20240229',
        api_key='sk-ant-test123',
        max_tokens=1024,
        temperature=0.7
    )

@pytest.fixture
def chain_execution():
    return ChainExecution.objects.create(
        chain_type='TestChain',
        status='running',
        metadata={'test': 'data'}
    )

@pytest.fixture
def chain_step(chain_execution):
    return ChainStep.objects.create(
        chain_execution=chain_execution,
        step_type='test_step',
        order=1,
        input_data={'input': 'test'},
        output_data={'output': 'test'}
    )

@pytest.fixture
def token_usage_log(chain_step, llm_provider):
    return TokenUsageLog.objects.create(
        chain_step=chain_step,
        provider=llm_provider,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        estimated_cost=0.002
    )

@pytest.fixture
def stored_artifact(chain_execution, chain_step):
    return StoredArtifact.objects.create(
        chain_execution=chain_execution,
        step=chain_step,
        name='test_artifact',
        data={'test': 'data'}
    ) 