import pytest
from src.django_llm.models import StoredArtifact

@pytest.mark.django_db
class TestStoredArtifact:
    def test_create_artifact(self, stored_artifact):
        assert stored_artifact.name == 'test_artifact'
        assert stored_artifact.data == {'test': 'data'}

    def test_chain_relationship(self, stored_artifact, chain_execution):
        assert stored_artifact.chain_execution == chain_execution
        assert stored_artifact in chain_execution.artifacts.all()

    def test_step_relationship(self, stored_artifact, chain_step):
        assert stored_artifact.step == chain_step
        assert stored_artifact in chain_step.artifacts.all()

    def test_artifact_without_step(self, chain_execution):
        artifact = StoredArtifact.objects.create(
            chain_execution=chain_execution,
            name='no_step_artifact',
            data={'test': 'data'}
        )
        assert artifact.step is None 