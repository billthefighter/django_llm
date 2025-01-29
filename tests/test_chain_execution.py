import pytest
from django.utils import timezone
from datetime import timedelta
from src.django_llm.models import ChainExecution

@pytest.mark.django_db
class TestChainExecution:
    def test_create_chain_execution(self, chain_execution):
        assert chain_execution.status == 'running'
        assert chain_execution.chain_type == 'TestChain'
        assert chain_execution.metadata == {'test': 'data'}

    def test_chain_completion(self, chain_execution):
        chain_execution.status = 'completed'
        chain_execution.completed_at = timezone.now()
        chain_execution.save()
        
        assert chain_execution.status == 'completed'
        assert chain_execution.completed_at is not None

    def test_chain_failure(self, chain_execution):
        error_msg = "Test error"
        chain_execution.status = 'failed'
        chain_execution.error_message = error_msg
        chain_execution.completed_at = timezone.now()
        chain_execution.save()
        
        assert chain_execution.status == 'failed'
        assert chain_execution.error_message == error_msg

    def test_chain_duration(self, chain_execution):
        start_time = timezone.now() - timedelta(seconds=10)
        chain_execution.started_at = start_time
        chain_execution.completed_at = timezone.now()
        chain_execution.save()
        
        duration = chain_execution.completed_at - chain_execution.started_at
        assert duration.total_seconds() >= 10 