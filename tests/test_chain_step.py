
import pytest
from django.utils import timezone

from src.django_llm.models import ChainStep


@pytest.mark.django_db
class TestChainStep:
    def test_create_chain_step(self, chain_step):
        assert chain_step.step_type == 'test_step'
        assert chain_step.order == 1
        assert chain_step.input_data == {'input': 'test'}
        assert chain_step.output_data == {'output': 'test'}

    def test_step_ordering(self, chain_execution):
        step1 = ChainStep.objects.create(
            chain_execution=chain_execution,
            step_type='step1',
            order=1
        )
        step2 = ChainStep.objects.create(
            chain_execution=chain_execution,
            step_type='step2',
            order=2
        )
        
        steps = chain_execution.steps.all()
        assert list(steps) == [step1, step2]

    def test_step_completion(self, chain_step):
        chain_step.completed_at = timezone.now()
        chain_step.save()
        
        assert chain_step.completed_at is not None

    def test_step_error(self, chain_step):
        error_msg = "Test error"
        chain_step.error_message = error_msg
        chain_step.save()
        
        assert chain_step.error_message == error_msg 