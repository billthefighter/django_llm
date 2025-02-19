
import redis
from channels.layers import get_channel_layer
from django.conf import settings


class RedisConfigChecker:
    @staticmethod
    async def check_channels_connection():
        """Check if Channels can connect to Redis"""
        try:
            channel_layer = get_channel_layer()
            await channel_layer.group_add("test_group", "test_channel")
            await channel_layer.group_discard("test_group", "test_channel")
            return True, "Channels Redis connection successful"
        except Exception as e:
            return False, f"Channels Redis connection failed: {str(e)}"

    @staticmethod
    def check_redis_direct():
        """Check direct Redis connection and memory"""
        try:
            # Get Redis connection info from settings
            redis_config = settings.CHANNEL_LAYERS['default']['CONFIG']['hosts'][0]
            host, port = redis_config

            # Create Redis connection
            r = redis.Redis(host=host, port=port)
            
            # Test connection
            r.ping()
            
            # Get Redis info
            info = r.info()
            
            # Check memory usage
            used_memory = info['used_memory_human']
            max_memory = info.get('maxmemory_human', 'unlimited')
            
            return True, {
                'status': 'Connected',
                'version': info['redis_version'],
                'used_memory': used_memory,
                'max_memory': max_memory,
                'connected_clients': info['connected_clients']
            }
        except redis.ConnectionError as e:
            return False, f"Redis connection failed: {str(e)}"
        except Exception as e:
            return False, f"Redis check failed: {str(e)}"

    @classmethod
    async def check_all(cls):
        """Run all Redis checks"""
        channels_status, channels_message = await cls.check_channels_connection()
        redis_status, redis_info = cls.check_redis_direct()
        
        return {
            'channels_connection': {
                'status': channels_status,
                'message': channels_message
            },
            'redis_connection': {
                'status': redis_status,
                'info': redis_info
            }
        } 