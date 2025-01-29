from django.utils import timezone
from datetime import timedelta

class ChainTestMixin:
    def create_complete_chain(self, chain_execution, steps=3):
        """Create a complete chain with multiple steps"""
        chain_execution.status = 'completed'
        chain_execution.completed_at = timezone.now()
        chain_execution.save()
        
        steps = [
            self.create_complete_step(chain_execution, i)
            for i in range(steps)
        ]
        return chain_execution, steps
    
    def create_complete_step(self, chain_execution, order):
        """Create a completed step with usage logs"""
        step = ChainStepFactory(
            chain_execution=chain_execution,
            order=order,
            completed_at=timezone.now()
        )
        TokenUsageLogFactory(chain_step=step)
        return step

class TimeMixin:
    def assert_recent(self, timestamp, seconds=5):
        """Assert that a timestamp is within the last few seconds"""
        now = timezone.now()
        diff = now - timestamp
        assert diff < timedelta(seconds=seconds) 