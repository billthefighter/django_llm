from django.test import Client
from django.urls import reverse


class LLMTestClient:
    def __init__(self):
        self.client = Client()
        
    def create_provider(self, data):
        """Create a provider through the admin interface"""
        return self.client.post(reverse('admin:django_llm_llmprovider_add'), data)
        
    def get_chain_execution(self, chain_id):
        """Get chain execution details"""
        return self.client.get(reverse('admin:django_llm_chainexecution_change', args=[chain_id]))
        
    def get_usage_stats(self, provider_id):
        """Get usage statistics for a provider"""
        return self.client.get(reverse('admin:django_llm_llmprovider_usage', args=[provider_id])) 