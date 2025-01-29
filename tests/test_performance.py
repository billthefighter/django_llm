import pytest
from django.test import override_settings
from .factories import ChainExecutionFactory, ChainStepFactory
from src.django_llm.models import ChainExecution

@pytest.mark.slow
class TestPerformance:
    def test_chain_query_performance(self, django_assert_num_queries):
        # Create test data
        chain = ChainExecutionFactory()
        steps = [ChainStepFactory(chain_execution=chain) for _ in range(10)]
        
        # Test query count
        with django_assert_num_queries(2):  # Should only need 2 queries
            chain = ChainExecution.objects.select_related(
                'steps'
            ).prefetch_related(
                'steps__token_usage'
            ).get(id=chain.id)
            steps = list(chain.steps.all()) 