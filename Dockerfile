# Ultra Simple Dockerfile - Run as root to debug
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install requirements
RUN pip install --upgrade pip
RUN pip install discord.py openai python-dotenv requests

# Verify installation
RUN python -c "import discord; print('Discord.py installed:', discord.__version__)"
RUN python -c "import openai; print('OpenAI installed')"

# Run bot as root (temporary for debugging)
CMD ["python", "bot.py"]
