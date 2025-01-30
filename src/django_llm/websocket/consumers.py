from channels.generic.websocket import AsyncJsonWebsocketConsumer
from asgiref.sync import sync_to_async
from django.core.serializers.json import DjangoJSONEncoder
import json

class ChainMonitorConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        # Extract chain_id from URL route
        self.chain_id = self.scope['url_route']['kwargs']['chain_id']
        self.room_group_name = f'chain_{self.chain_id}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Handle chain execution updates
    async def chain_update(self, event):
        """Send chain execution updates to WebSocket"""
        await self.send(text_data=json.dumps(
            event['data'],
            cls=DjangoJSONEncoder
        )) 