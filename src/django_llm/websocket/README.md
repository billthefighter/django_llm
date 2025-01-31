# Django LLM WebSocket Monitoring

Real-time monitoring for LLM chain executions using WebSockets. This feature enables live updates during chain execution, including progress tracking, intermediate results, and error handling.

## Setup

1. Install required dependencies:
```bash
pip install channels channels-redis
```

2. Configure Redis (required for channel layers):
```bash
# Install Redis server
sudo apt-get install redis-server  # Ubuntu/Debian
brew install redis                 # macOS
```

3. Update your Django settings:
```python
INSTALLED_APPS = [
    ...
    'channels',
]

ASGI_APPLICATION = 'config.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}
```

4. Include WebSocket URLs in your ASGI configuration:
```python
# config/asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django_llm.websocket import routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns
        )
    ),
})
```

## Usage

### Backend Implementation

1. Add WebSocket support to your chain:
```python
from django_llm.chains import SequentialChain
from django_llm.websocket.mixins import WebSocketChainMixin

class MyChain(WebSocketChainMixin, SequentialChain):
    pass
```

2. Create chain instance and execute:
```python
chain = MyChain(llm=llm, steps=[...])
result = await chain.execute(input_data="Your input")
```

### Frontend Implementation

1. Include the required JavaScript files:
```html
<script src="{% static 'js/chain-monitor.js' %}"></script>
```

2. Basic monitoring setup:
```javascript
const monitor = new ChainMonitor('your-chain-id');

monitor.onStart((data) => {
    console.log('Chain started:', data);
    showLoadingIndicator();
});

monitor.onStep((data) => {
    console.log('Step completed:', data);
    updateProgress(data);
});

monitor.onComplete((data) => {
    console.log('Chain completed:', data);
    hideLoadingIndicator();
    showResult(data.result);
});

monitor.onError((data) => {
    console.error('Chain error:', data);
    showErrorMessage(data.error);
});
```

3. Full example with UI updates:
See `frontend_example.js` for a complete implementation with UI handling.

## WebSocket Events

The following events are emitted during chain execution:

### Start Event
```javascript
{
    update_type: 'start',
    status: 'started',
    metadata: { /* chain metadata */ }
}
```

### Step Event
```javascript
{
    update_type: 'step',
    step_name: 'step_name',
    progress_percentage: 50,
    intermediate_result: { /* step result */ }
}
```

### Complete Event
```javascript
{
    update_type: 'complete',
    status: 'completed',
    result: { /* final result */ }
}
```

### Error Event
```javascript
{
    update_type: 'error',
    status: 'error',
    error: 'Error message'
}
```

## Security Considerations

1. Authentication: The WebSocket consumer uses Django's authentication system. Ensure proper authentication is implemented for your WebSocket connections.

2. Input Validation: Always validate chain IDs and other user inputs before processing.

3. Rate Limiting: Consider implementing rate limiting for WebSocket connections if needed.

## Troubleshooting

1. Connection Issues:
   - Ensure Redis is running
   - Check ASGI configuration
   - Verify WebSocket URL format

2. Missing Updates:
   - Confirm chain is using WebSocketChainMixin
   - Check channel layer configuration
   - Verify chain_id matches between frontend and backend

3. Redis Errors:
   - Check Redis server status
   - Verify Redis connection settings
   - Ensure sufficient memory is available

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This feature is part of Django LLM and is licensed under the MIT License.

# Redis Setup and Configuration

## 1. Installation

### Ubuntu/Debian:
```bash
# Install Redis
sudo apt-get update
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### macOS:
```bash
# Install Redis using Homebrew
brew install redis

# Start Redis service
brew services start redis
```

## 2. Verify Installation

Run the Redis checker command:
```bash
python manage.py check_redis
```

This will:
- Check Redis connection status
- Verify Channels integration
- Display memory usage and version info
- Provide troubleshooting recommendations if needed

## 3. Redis Configuration

### Basic Configuration
Edit Redis configuration file (usually at `/etc/redis/redis.conf`):

```conf
# Basic settings
port 6379
bind 127.0.0.1
maxmemory 256mb
maxmemory-policy allkeys-lru

# Recommended for Django Channels
timeout 0
tcp-keepalive 60
```

### Security Settings
```conf
# Enable password protection
requirepass your_strong_password

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
```

If you set a password, update your Django settings:
```python
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379, {'password': 'your_strong_password'})],
        },
    },
}
```

## 4. Monitoring

### Check Redis Status
```bash
# Check service status
sudo systemctl status redis-server  # Ubuntu/Debian
brew services list                 # macOS

# Monitor Redis
redis-cli monitor

# Check memory usage
redis-cli info memory
```

### Common Issues

1. Connection Refused:
   ```bash
   sudo systemctl restart redis-server
   ```

2. Memory Issues:
   ```bash
   # Check memory usage
   redis-cli info memory
   
   # Clear all data if needed
   redis-cli flushall
   ```

3. Permission Issues:
   ```bash
   # Check Redis logs
   sudo tail -f /var/log/redis/redis-server.log
   
   # Fix permissions
   sudo chown redis:redis /var/lib/redis
   sudo chmod 750 /var/lib/redis
   ```

## 5. Performance Tuning

For production environments:

```conf
# Memory optimization
maxmemory-samples 10
activerehashing yes
hz 100

# Persistence settings
save 900 1
save 300 10
save 60 10000

# Client settings
maxclients 10000
timeout 300
```

## 6. Backup and Recovery

Create automated backup script:
```bash
#!/bin/bash
BACKUP_DIR="/var/lib/redis/backups"
DATE=$(date +%Y%m%d_%H%M%S)
redis-cli SAVE
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis_backup_$DATE.rdb"
find "$BACKUP_DIR" -name "redis_backup_*" -mtime +7 -delete
``` 