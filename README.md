# Django LLM

Django LLM is a Django application that provides seamless integration between Django projects and LLM (Large Language Model) applications. It offers robust storage, tracking, and management of LLM interactions within your Django application.

Built on top of [LLM Orchestrator](https://github.com/billthefighter/llm_orchestrator), a powerful system for managing large-scale LLM tasks through decomposition and parallel processing.

## Features

- 🔄 **Chain Execution Tracking**: Track and monitor all LLM chain executions in your Django database
- 📊 **Token Usage Monitoring**: Track token usage and costs across different LLM providers
- 💾 **Artifact Storage**: Store chain artifacts directly in your Django database
- 🔍 **Step-by-Step Tracking**: Monitor individual steps within chain executions
- 🔌 **Provider Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- 📈 **Performance Metrics**: Track execution times, token usage, and costs

## Installation

1. Install the package using pip:

```bash
pip install django-llm
```

2. Add 'django_llm' to your INSTALLED_APPS in settings.py:

```python
INSTALLED_APPS = [
    ...
    'django_llm',
]
```

3. Run migrations:

```bash
python manage.py migrate
```

## Quick Start

```python
from django_llm.chains import SequentialChain
from llm_orchestrator.src.llm.base import BaseLLMInterface

# Initialize your LLM interface
llm = BaseLLMInterface(config)

# Create and execute a chain
chain = SequentialChain(
    llm=llm,
    steps=[...]
)

# Execute the chain - results and artifacts will be automatically stored
result = await chain.execute(input_data="Your input")
```

## Models

### LLMProvider
Manages LLM provider configurations:
- Provider name and model selection
- API keys and authentication
- Model parameters (max tokens, temperature)
- Configuration validation
- Usage tracking

```python
# Create a provider configuration
provider = LLMProvider.objects.create(
    name='anthropic',
    model='claude-3-sonnet-20240229',
    api_key='sk-ant-xxxx',
    max_tokens=1024,
    temperature=0.7
)

# Get configuration for LLM Orchestrator
config = provider.get_config_dict()

# Initialize LLM with config
llm = BaseLLMInterface(config)
```

### ChainExecution
Tracks overall chain executions:
- Chain type
- Execution status
- Start/completion times
- Metadata
- Error tracking

### ChainStep
Tracks individual steps within a chain:
- Input/output data
- Step type
- Execution order
- Timing information

### TokenUsageLog
Monitors token usage and costs:
- Token counts (prompt/completion)
- Cost estimates
- Provider information
- Model details

### StoredArtifact
Stores chain artifacts:
- JSON-serializable data
- Associated chain/step
- Creation timestamp

## Advanced Usage

### Creating a Custom Chain

```python
from django_llm.chains import SequentialChain
from llm_orchestrator.src.llm.base import ChainStep

class CustomChain(SequentialChain):
    def __init__(self, llm):
        steps = [
            ChainStep(
                task_type="summarize",
                input_transform=lambda ctx, **kwargs: {"text": kwargs["input_text"]},
            ),
            ChainStep(
                task_type="analyze",
                input_transform=lambda ctx, **kwargs: {"summary": kwargs["previous_result"]},
            )
        ]
        super().__init__(llm=llm, steps=steps)
```

### Querying Chain History

```python
from django_llm.models import ChainExecution, TokenUsageLog
from django.db.models import Sum

# Get all successful chain executions
successful_chains = ChainExecution.objects.filter(status='completed')

# Get token usage statistics
total_tokens = TokenUsageLog.objects.aggregate(
    total_tokens=Sum('total_tokens'),
    total_cost=Sum('estimated_cost')
)
```

## Configuration

Django LLM can be configured through Django settings:

```python
DJANGO_LLM = {
    'DEFAULT_PROVIDER': 'provider_name',  # References LLMProvider.name
    'STORAGE_BACKEND': 'django_llm.storage.DatabaseStorage',
    'TOKEN_TRACKING': True,
    'COST_TRACKING': True,
}
```

### Provider Management

Providers can be managed through Django admin or programmatically:

```python
from django_llm.models import LLMProvider

# List available models for a provider
available_models = LLMProvider.get_available_models('anthropic')

# Create a new provider configuration
provider = LLMProvider.objects.create(
    name='anthropic',
    model='claude-3-sonnet-20240229',
    api_key='sk-ant-xxxx',
    max_tokens=1024,
    temperature=0.7,
    config={
        'additional_setting': 'value'
    }
)

# Get active provider configuration
active_provider = LLMProvider.objects.filter(
    is_active=True,
    name='anthropic'
).first()

# Track usage
usage_stats = active_provider.usage_logs.aggregate(
    total_tokens=Sum('total_tokens'),
    total_cost=Sum('estimated_cost')
)
```

## Architecture

Django LLM integrates with the LLM Orchestrator library to provide:

1. **Database Storage**: Replace file-based storage with Django's ORM
2. **Execution Tracking**: Automatic tracking of all chain executions
3. **Token Usage**: Monitor and track token usage across providers
4. **Artifact Management**: Store and retrieve chain artifacts

The package uses:
- Django models for persistent storage
- Async execution for chain processing
- Mixins for easy integration
- Provider-agnostic interfaces

## Development

To set up for development:

1. Clone the repository with submodules:

```bash
git clone --recursive https://github.com/yourusername/django-llm.git
cd django-llm
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Run tests:

```bash
poetry run pytest
```

## Roadmap

### 1. WebSocket Support for Real-time Chain Monitoring (In Progress)
Our top priority is implementing WebSocket support to provide real-time visibility into LLM chain execution. This will enable:
- Live monitoring of chain execution steps
- Real-time feedback for loading/typing indicators
- Immediate visibility into token usage and costs
- Better error handling and debugging capabilities

This foundation will support many future features and improvements.

#### Next Steps for WebSocket Implementation:
1. Set up Django Channels infrastructure
2. Create WebSocket consumers for chain monitoring
3. Implement middleware for tracking chain execution
4. Add client-side handlers for WebSocket events
5. Create documentation and examples

### Future Priorities

2. **Streaming Responses**
- Building on WebSocket infrastructure
- Implementing streaming handlers
- Adding progress indicators
- Creating streaming-compatible templates

3. **Rate Limiting and Quota Management**
- Implementation of rate limiting middleware
- Usage tracking and quota enforcement
- Admin interface for quota management
- Alert system for quota limits

4. **Pre-built UI Components**
- Common LLM interface patterns
- React/Vue.js components
- Customizable templates
- Mobile-responsive designs

5. **Provider-Specific Examples**
- Integration guides for popular LLM providers
- Best practices documentation
- Performance optimization tips
- Common use-case examples

## Contributing

We welcome contributions to any of these roadmap items! Please check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Created by Lucas Whipple (lucas.whipple@gmail.com)

Built on top of the [LLM Orchestrator](https://github.com/billthefighter/llm_orchestrator) library.

## Similar Projects

While Django LLM aims to be the most comprehensive solution for enterprise-grade LLM applications, there are other excellent projects in this space:

- [LangChain Django](https://github.com/jacobsvante/langchain-django) - Focused on LangChain integration
- [Django LLM Tools](https://github.com/kreneskyp/django-llm-tools) - Lightweight LLM integration
- [Django GPT Logger](https://github.com/richardcornish/django-gpt-logger) - OpenAI-specific logging solution

See [what's_the_point.md](what's_the_point.md) for a detailed comparison and when to choose Django LLM.
