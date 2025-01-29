import factory
from factory.django import DjangoModelFactory
from src.django_llm.models import LLMProvider, ChainExecution, ChainStep, TokenUsageLog, StoredArtifact

class LLMProviderFactory(DjangoModelFactory):
    class Meta:
        model = LLMProvider

    name = 'anthropic'
    model = 'claude-3-sonnet-20240229'
    api_key = factory.Sequence(lambda n: f'sk-ant-test{n}')
    max_tokens = 1024
    temperature = 0.7
    is_active = True

class ChainExecutionFactory(DjangoModelFactory):
    class Meta:
        model = ChainExecution

    chain_type = factory.Sequence(lambda n: f'TestChain{n}')
    status = 'running'
    metadata = factory.Dict({'test': 'data'})

class ChainStepFactory(DjangoModelFactory):
    class Meta:
        model = ChainStep

    chain_execution = factory.SubFactory(ChainExecutionFactory)
    step_type = factory.Sequence(lambda n: f'test_step_{n}')
    order = factory.Sequence(lambda n: n)
    input_data = factory.Dict({'input': 'test'})
    output_data = factory.Dict({'output': 'test'})

class TokenUsageLogFactory(DjangoModelFactory):
    class Meta:
        model = TokenUsageLog

    chain_step = factory.SubFactory(ChainStepFactory)
    provider = factory.SubFactory(LLMProviderFactory)
    prompt_tokens = 100
    completion_tokens = 50
    total_tokens = 150
    estimated_cost = 0.002

class StoredArtifactFactory(DjangoModelFactory):
    class Meta:
        model = StoredArtifact

    chain_execution = factory.SubFactory(ChainExecutionFactory)
    step = factory.SubFactory(ChainStepFactory)
    name = factory.Sequence(lambda n: f'test_artifact_{n}')
    data = factory.Dict({'test': 'data'}) 