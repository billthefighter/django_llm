from django.core.management.base import BaseCommand
import asyncio
from django_llm.utils.redis_checker import RedisConfigChecker
#from rich.console import Console
#from rich.table import Table

class Command(BaseCommand):
    help = 'Check Redis configuration and connection status'

    def handle(self, *args, **options):
        console = Console()

        async def run_checks():
            with console.status("[bold green]Checking Redis configuration..."):
                results = await RedisConfigChecker.check_all()
            
            # Display results in a table
            table = Table(title="Redis Configuration Check Results")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="yellow")

            # Channels results
            channels = results['channels_connection']
            table.add_row(
                "Channels Connection",
                "✅ OK" if channels['status'] else "❌ Failed",
                channels['message']
            )

            # Redis connection results
            redis_conn = results['redis_connection']
            if redis_conn['status']:
                info = redis_conn['info']
                table.add_row(
                    "Redis Connection",
                    "✅ OK",
                    f"Version: {info['version']}\n"
                    f"Memory: {info['used_memory']} / {info['max_memory']}\n"
                    f"Clients: {info['connected_clients']}"
                )
            else:
                table.add_row(
                    "Redis Connection",
                    "❌ Failed",
                    str(redis_conn['info'])
                )

            console.print(table)

            # Print recommendations if there are issues
            if not all(r['status'] for r in results.values()):
                console.print("\n[bold red]Troubleshooting Recommendations:[/bold red]")
                console.print("""
1. Check if Redis is running:
   - Ubuntu/Debian: sudo systemctl status redis
   - macOS: brew services list

2. Verify Redis installation:
   - Ubuntu/Debian: sudo apt-get install redis-server
   - macOS: brew install redis

3. Check Redis configuration:
   - Configuration file location: /etc/redis/redis.conf
   - Default port: 6379
   - Bind address: Usually 127.0.0.1

4. Verify Django settings:
   - Check CHANNEL_LAYERS configuration
   - Verify Redis host and port

5. Common fixes:
   - Restart Redis: sudo service redis restart
   - Clear Redis: redis-cli flushall
   - Check Redis logs: sudo tail -f /var/log/redis/redis-server.log
                """)

        asyncio.run(run_checks()) 