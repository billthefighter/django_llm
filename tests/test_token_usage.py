from decimal import Decimal

import pytest


@pytest.mark.django_db
class TestTokenUsageLog:
    def test_create_token_usage(self, token_usage_log):
        assert token_usage_log.prompt_tokens == 100
        assert token_usage_log.completion_tokens == 50
        assert token_usage_log.total_tokens == 150
        assert token_usage_log.estimated_cost == Decimal('0.002')

    def test_provider_relationship(self, token_usage_log, llm_provider):
        assert token_usage_log.provider == llm_provider
        assert token_usage_log in llm_provider.usage_logs.all()

    def test_chain_step_relationship(self, token_usage_log, chain_step):
        assert token_usage_log.chain_step == chain_step
        assert token_usage_log in chain_step.token_usage.all() 