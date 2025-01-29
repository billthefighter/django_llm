import pytest
from django.core.exceptions import ValidationError
from src.django_llm.models import LLMProvider

@pytest.mark.django_db
class TestLLMProvider:
    def test_create_provider(self, llm_provider):
        assert llm_provider.name == 'anthropic'
        assert llm_provider.model == 'claude-3-sonnet-20240229'
        assert llm_provider.is_active is True

    def test_get_config_dict(self, llm_provider):
        config = llm_provider.get_config_dict()
        assert config['llm']['provider'] == 'anthropic'
        assert config['llm']['model'] == 'claude-3-sonnet-20240229'
        assert config['llm']['max_tokens'] == 1024
        assert config['llm']['temperature'] == 0.7

    def test_validate_model_success(self, llm_provider):
        assert llm_provider.validate_model() is True

    def test_validate_model_failure(self):
        with pytest.raises(ValueError):
            LLMProvider.objects.create(
                name='anthropic',
                model='invalid-model',
                api_key='sk-ant-test123'
            )

    def test_get_available_models(self):
        models = LLMProvider.get_available_models('anthropic')
        assert 'claude-3-sonnet-20240229' in models
        assert len(models) == 3

    def test_total_usage_calculation(self, llm_provider, token_usage_log):
        total = llm_provider.usage_logs.aggregate(
            total_tokens=models.Sum('total_tokens')
        )['total_tokens']
        assert total == 150 