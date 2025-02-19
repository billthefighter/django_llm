import pytest

from src.django_llm.models import ChainExecution

from .factories import ChainExecutionFactory, ChainStepFactory


@pytest.mark.slow
class TestPerformance:
    def test_chain_query_performance(self, django_assert_num_queries):
        # Create test data
        chain = ChainExecutionFactory()
        [ChainStepFactory(chain_execution=chain) for _ in range(10)]
        
        # Test query count
        with django_assert_num_queries(2):  # Should only need 2 queries
            chain = ChainExecution.objects.select_related(
                'steps'
            ).prefetch_related(
                'steps__token_usage'
            ).get(id=chain.id)
            list(chain.steps.all()) 