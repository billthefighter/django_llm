from submodules.llm_orchestrator.src.llm.chains import SequentialChain as BaseSequentialChain
from .mixins import DjangoChainMixin

class SequentialChain(DjangoChainMixin, BaseSequentialChain):
    """Django-enabled Sequential Chain"""
    pass 