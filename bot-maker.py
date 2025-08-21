import os

config_dir = 'config'

personas = [name for name in os.listdir(config_dir) if os.path.isdir(os.path.join(config_dir, name))]

with open('docker-compose.yaml', 'w') as f:
    f.write('services:
')
    for p in sorted(personas):
        service = f'''  {p}:
    build:
      context: .  # Build from root context
      dockerfile: Dockerfile  # Use shared root Dockerfile
    container_name: {p}-bot
    restart: unless-stopped
    env_file:
      - {config_dir}/{p}/.env  # Load env from persona config dir
    volumes:
      - ./logs/{p}:/app/logs  # Isolated logs for {p}
'''
        f.write(service)

print('✓ Generated docker-compose.yaml with services for: ' + ', '.join(personas))
