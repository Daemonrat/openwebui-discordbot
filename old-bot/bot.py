# Fixed Bot Code - Compatible with discord.py
# Removed slash commands and other incompatible features

import os
import logging
import asyncio
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import sys
import concurrent.futures

import discord
from discord.ext import commands, tasks
from openai import OpenAI
import base64
import requests
from io import BytesIO
from collections import deque
from dotenv import load_dotenv

# Configure proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration with validation
class BotConfig:
    def __init__(self):
        self.DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  
        self.OPENWEBUI_API_BASE = os.getenv('OPENWEBUI_API_BASE')
        self.MODEL_NAME = os.getenv('MODEL_NAME')
        
        # Validate required environment variables
        self._validate_config()
    
    def _validate_config(self):
        required_vars = {
            'DISCORD_TOKEN': self.DISCORD_TOKEN,
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
            'OPENWEBUI_API_BASE': self.OPENWEBUI_API_BASE,
            'MODEL_NAME': self.MODEL_NAME
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Ensure HTTPS for API base URL (warn only)
        if self.OPENWEBUI_API_BASE and not self.OPENWEBUI_API_BASE.startswith('https://'):
            if self.OPENWEBUI_API_BASE.startswith('http://localhost') or self.OPENWEBUI_API_BASE.startswith('http://127.0.0.1'):
                logger.info("Using HTTP for localhost - acceptable for development")
            else:
                logger.warning("API base URL should use HTTPS for security")

config = BotConfig()

# Enhanced OpenAI client configuration
try:
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENWEBUI_API_BASE
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Enhanced Discord bot configuration
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(
    command_prefix='!',
    intents=intents,
    case_insensitive=True
)

# Enhanced message history management with size limits
class MessageHistoryManager:
    def __init__(self, max_channels: int = 100, max_messages_per_channel: int = 50):
        self.channel_history: Dict[int, deque] = {}
        self.max_channels = max_channels
        self.max_messages_per_channel = max_messages_per_channel
    
    def add_message(self, channel_id: int, message: str):
        if channel_id not in self.channel_history:
            self.channel_history[channel_id] = deque(maxlen=self.max_messages_per_channel)
        
        self.channel_history[channel_id].append(message)
        
        # Simple cleanup if too many channels
        if len(self.channel_history) > self.max_channels:
            oldest_channel = next(iter(self.channel_history))
            del self.channel_history[oldest_channel]
    
    def get_history(self, channel_id: int, limit: int = None) -> str:
        if channel_id not in self.channel_history:
            return ""
        
        messages = list(self.channel_history[channel_id])
        if limit:
            messages = messages[-limit:]
        
        return "\n".join(messages)

history_manager = MessageHistoryManager()

async def run_blocking_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: fn(*args, **kwargs))
    return result

# Enhanced image download with better error handling
async def download_image(url: str, max_size_mb: int = 10) -> Optional[str]:
    try:
        response = requests.get(url, timeout=10, stream=True)
        if response.status_code != 200:
            logger.warning(f"Failed to download image: HTTP {response.status_code}")
            return None
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            logger.warning(f"Image too large: {content_length} bytes")
            return None
        
        image_data = BytesIO()
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            if downloaded > max_size_mb * 1024 * 1024:
                logger.warning(f"Downloaded image too large: {downloaded} bytes")
                return None
            image_data.write(chunk)
        
        image_data.seek(0)
        base64_image = base64.b64encode(image_data.read()).decode('utf-8')
        logger.info(f"Successfully downloaded and encoded image from {url}")
        return base64_image
        
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

async def get_chat_history(channel, limit: int = 50) -> str:
    try:
        if limit > 100:
            limit = 100
        
        messages = []
        async for message in channel.history(limit=limit):
            if message.author.bot:
                continue
                
            content = f"{message.author.display_name}: {message.content}"
            
            for attachment in message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                    if attachment.size < 10 * 1024 * 1024:
                        content += f" [Image: {attachment.url}]"
                    else:
                        content += f" [Large Image: {attachment.filename}]"
            
            messages.append(content)
        
        history_text = "\n".join(reversed(messages))
        history_manager.add_message(channel.id, history_text)
        return history_text
        
    except Exception as e:
        logger.error(f"Error fetching message history: {e}")
        return "Error fetching message history"

async def get_ai_response(context: str, user_message: str, image_urls: Optional[List[str]] = None, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": []}]
            text_content = f"##CONTEXT##\n{context}\n##ENDCONTEXT##\n\n{user_message}"
            messages[0]["content"].append({"type": "text", "text": text_content})
            if image_urls:
                for url in image_urls[:3]:
                    base64_image = await download_image(url)
                    if base64_image:
                        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            response = await run_blocking_in_thread(
                client.chat.completions.create,
                model=config.MODEL_NAME,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            result = response.choices[0].message.content
            logger.info(f"Successfully got AI response (attempt {attempt + 1})")
            return result
        except Exception as e:
            logger.error(f"AI request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return "I'm sorry, I'm having trouble connecting to the AI service right now. Please try again later."

# Event handlers
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guilds')
    
    # Start background tasks
    cleanup_history.start()
    
    # Update bot status
    try:
        await bot.change_presence(
            status=discord.Status.online,
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="your messages | Powered by Daemonrat"
            )
        )
    except Exception as e:
        logger.error(f"Failed to set bot status: {e}")

@bot.event
async def on_error(event, *args, **kwargs):
    logger.error(f"Error in event {event}")

@bot.event  
async def on_message(message):
    # Ignore messages from bots
    if message.author.bot:
        return
    
    should_respond = False
    
    # Check if bot was mentioned
    if bot.user in message.mentions:
        should_respond = True
    
    # Check if message is a DM
    if isinstance(message.channel, discord.DMChannel):
        should_respond = True
    
    if should_respond:
        # Input validation
        if len(message.content) > 2000:
            await message.reply("Your message is too long. Please keep it under 2000 characters.")
            return
        
        async with message.channel.typing():
            try:
                # Get chat history
                history = await get_chat_history(message.channel, limit=20)
                
                # Remove bot mention from the message
                user_message = message.content.replace(f'<@{bot.user.id}>', '').strip()
                
                # Collect image URLs from the message
                image_urls = []
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        if attachment.size < 10 * 1024 * 1024:
                            image_urls.append(attachment.url)
                
                # Get AI response
                response = await get_ai_response(history, user_message, image_urls)
                
                # Split long responses
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    for chunk in chunks:
                        await message.reply(chunk)
                else:
                    await message.reply(response)
                    
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                try:
                    await message.reply("An error occurred while processing your request. Please try again.")
                except:
                    pass
    
    # Process commands
    await bot.process_commands(message)

# Background task for cleanup
@tasks.loop(hours=1)
async def cleanup_history():
    logger.info("Running periodic cleanup...")

# Regular commands (not slash commands)
@bot.command(name="ping")
async def ping(ctx):
    """Check if the bot is responsive"""
    latency = round(bot.latency * 1000)
    await ctx.send(f" Pong! Latency: {latency}ms")

@bot.command(name="status")
async def status(ctx):
    """Get bot status information"""
    embed = discord.Embed(
        title="Bot Status",
        color=discord.Color.green(),
        timestamp=datetime.now()
    )
    
    embed.add_field(name="Guilds", value=len(bot.guilds), inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Model", value=config.MODEL_NAME, inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="help_ai")
async def help_ai(ctx):
    """Show help information about the AI bot"""
    embed = discord.Embed(
        title="OpenWebUI Discord Bot Help",
        description="I'm an AI assistant powered by OpenWebUI. Here's how to interact with me:",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name=" Chat with me",
        value=" Mention me in any message: `@botname your question`\n Send me a DM\n I can process images you attach to your messages",
        inline=False
    )
    
    embed.add_field(
        name=" Commands",
        value=" `!ping` - Check if I'm responsive\n `!status` - View bot information\n `!help_ai` - Show this help message",
        inline=False
    )
    
    embed.add_field(
        name=" Image Support",
        value="Attach images to your messages and I'll analyze them along with your text!",
        inline=False
    )
    
    await ctx.send(embed=embed)

# Function to list available tools from OpenWebUI
def get_tools_list():
    try:
        response = requests.get(f"{OPENWEBUI_API_BASE}/tools", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return str(e)

# Function to enable a tool
def enable_tool(tool_id):
    try:
        response = requests.post(f"{OPENWEBUI_API_BASE}/tools/{tool_id}/enable", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
        if response.status_code == 200:
            return "Tool enabled successfully."
        else:
            return f"Failed to enable tool: {response.status_code}"
    except Exception as e:
        return str(e)

# Function to disable a tool
def disable_tool(tool_id):
    try:
        response = requests.post(f"{OPENWEBUI_API_BASE}/tools/{tool_id}/disable", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
        if response.status_code == 200:
            return "Tool disabled successfully."
        else:
            return f"Failed to disable tool: {response.status_code}"
    except Exception as e:
        return str(e)

# Discord command to list tools
@bot.command(name='tools-list', help='Lists available tools from OpenWebUI')
async def tools_list(ctx):
    tools = get_tools_list()
    if isinstance(tools, str):  # Error message
        await ctx.send(f"Error fetching tools: {tools}")
    elif tools:
        response = "Available Tools:\n"
        for tool in tools:
            tool_id = tool.get('id', 'Unknown ID')
            tool_name = tool.get('name', 'Unknown Name')
            tool_status = 'Enabled' if tool.get('enabled', False) else 'Disabled'
            response += f"- ID: {tool_id}, Name: {tool_name}, Status: {tool_status}\n"
        await ctx.send(response)
    else:
        await ctx.send("No tools available or failed to fetch tools.")

# Discord command to enable a tool
@bot.command(name='tools-enable', help='Enables a tool by ID')
async def tools_enable(ctx, tool_id):
    if not tool_id:
        await ctx.send("Please provide a tool ID. Usage: !tools-enable <tool_id>")
        return
    result = enable_tool(tool_id)
    await ctx.send(result)

# Discord command to disable a tool
@bot.command(name='tools-disable', help='Disables a tool by ID')
async def tools_disable(ctx, tool_id):
    if not tool_id:
        await ctx.send("Please provide a tool ID. Usage: !tools-disable <tool_id>")
        return
    result = disable_tool(tool_id)
    await ctx.send(result)

def main():
    try:
        logger.info("Starting Discord bot...")
        bot.run(config.DISCORD_TOKEN, log_handler=None)
        
    except discord.LoginFailure:
        logger.error("Invalid Discord token provided")
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
