
from channels.layers import get_channel_layer


class WebSocketChainMixin:
    async def _send_update(self, update_type, data):
        """Send update to WebSocket channel"""
        channel_layer = get_channel_layer()
        
        if not hasattr(self, 'chain_execution'):
            return
            
        room_group_name = f'chain_{self.chain_execution.id}'
        
        await channel_layer.group_send(
            room_group_name,
            {
                'type': 'chain_update',
                'data': {
                    'update_type': update_type,
                    **data
                }
            }
        )

    async def execute(self, *args, **kwargs):
        """Override execute to add WebSocket updates"""
        # Send start notification
        await self._send_update('start', {
            'status': 'started',
            'metadata': self.chain_execution.metadata
        })

        try:
            result = await super().execute(*args, **kwargs)
            
            # Send completion notification
            await self._send_update('complete', {
                'status': 'completed',
                'result': result
            })
            
            return result
            
        except Exception as e:
            # Send error notification
            await self._send_update('error', {
                'status': 'error',
                'error': str(e)
            })
            raise 