#!/bin/bash

# Run the bot-maker.py script to generate the docker-compose.yaml
python3 bot-maker.py

# Then start all docker-compose services
# You can change this to 'docker-compose' if needed depending on your environment
# Using --remove-orphans to clean up any services no longer defined

docker compose up -d --remove-orphans
