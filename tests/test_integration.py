import pytest
from django.urls import reverse

from .factories import ChainExecutionFactory


@pytest.mark.integration
class TestAdminIntegration:
    def test_provider_creation(self, admin_client):
        url = reverse('admin:django_llm_llmprovider_add')
        data = {
            'name': 'anthropic',
            'model': 'claude-3-sonnet-20240229',
            'api_key': 'sk-ant-test123',
            'max_tokens': 1024,
            'temperature': 0.7,
            'is_active': True
        }
        response = admin_client.post(url, data)
        assert response.status_code == 302  # Redirect after success

    def test_chain_execution_view(self, admin_client):
        chain = ChainExecutionFactory()
        url = reverse('admin:django_llm_chainexecution_change', args=[chain.id])
        response = admin_client.get(url)
        assert response.status_code == 200 