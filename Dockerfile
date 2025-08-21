# Ultra Simple Dockerfile - Run as root to debug
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/

# Copy configuration
COPY config/ ./config/

# Install requirements from centralized location
RUN pip install --upgrade pip
RUN pip install -r config/requirements.txt

# Verify installation
RUN python -c "import discord; print('Discord.py installed:', discord.__version__)"
RUN python -c "import openai; print('OpenAI installed')"

# Run bot from src directory
CMD ["python", "src/bot.py"]