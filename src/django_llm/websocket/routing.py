from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chain/(?P<chain_id>\w+)/$', consumers.ChainMonitorConsumer.as_asgi()),
] 