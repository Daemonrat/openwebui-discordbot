# Enhanced bot.py with OpenWebUI Tools API Workaround
# This version addresses the tools API limitations and adds MCP support

import os
import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import sys
import concurrent.futures
import json

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
        self.ENABLE_MCP = os.getenv('ENABLE_MCP', 'false').lower() == 'true'
        self.MCP_SERVER_URL = os.getenv('MCP_SERVER_URL')

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

# Enhanced OpenWebUI Tools Manager
class OpenWebUIToolsManager:
    def __init__(self, api_base: str, api_key: str):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.tools_loaded = False
        self.available_tools = []

    async def initialize_tools(self):
        """Initialize tools by calling the tools endpoint first (workaround for API limitation)"""
        try:
            # This is the workaround: call tools endpoint to load tools schema
            response = requests.get(
                f"{self.api_base}/api/v1/tools/",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )

            if response.status_code == 200:
                self.available_tools = response.json()
                self.tools_loaded = True
                logger.info(f"Successfully loaded {len(self.available_tools)} tools from OpenWebUI")
                return True
            else:
                logger.warning(f"Failed to load tools: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
            return False

    async def call_with_tools(self, messages: List[Dict], tool_ids: Optional[List[str]] = None):
        """Make API call with tools enabled (after initialization)"""
        if not self.tools_loaded:
            await self.initialize_tools()

        try:
            data = {
                "model": config.MODEL_NAME,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.7
            }

            # Add tool_ids if tools are available
            if self.tools_loaded and tool_ids:
                data["tool_ids"] = tool_ids
            elif self.tools_loaded and self.available_tools:
                # Auto-select available tools
                data["tool_ids"] = [tool.get("id") for tool in self.available_tools if tool.get("id")]

            response = requests.post(
                f"{self.api_base}/api/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API call failed: HTTP {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error calling API with tools: {e}")
            return None

# MCP Integration Class
class MCPIntegration:
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        self.enabled = server_url is not None

    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Call an MCP tool if available"""
        if not self.enabled:
            return None

        try:
            response = requests.post(
                f"{self.server_url}/tools/{tool_name}",
                json={"parameters": parameters},
                timeout=15
            )

            if response.status_code == 200:
                return response.json().get("result", "")
            else:
                logger.warning(f"MCP tool call failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return None

    async def list_tools(self) -> List[Dict]:
        """List available MCP tools"""
        if not self.enabled:
            return []

        try:
            response = requests.get(f"{self.server_url}/tools", timeout=10)
            if response.status_code == 200:
                return response.json().get("tools", [])
        except Exception as e:
            logger.error(f"Error listing MCP tools: {e}")

        return []

# Initialize managers
tools_manager = OpenWebUIToolsManager(config.OPENWEBUI_API_BASE, config.OPENAI_API_KEY)
mcp_integration = MCPIntegration(config.MCP_SERVER_URL if config.ENABLE_MCP else None)

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
    """Enhanced AI response with OpenWebUI tools and MCP support"""

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

            # Try OpenWebUI tools first
            if tools_manager.tools_loaded or await tools_manager.initialize_tools():
                logger.info("Using OpenWebUI tools for response")
                api_response = await tools_manager.call_with_tools(messages)
                if api_response and api_response.get("choices"):
                    return api_response["choices"][0]["message"]["content"]

            # Fallback to standard OpenAI client
            response = await run_blocking_in_thread(
                client.chat.completions.create,
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": text_content}],
                max_tokens=2000,
                temperature=0.7
            )

            result = response.choices[0].message.content
            logger.info(f"Successfully got AI response (attempt {attempt + 1})")

            # Try to enhance with MCP if available
            if mcp_integration.enabled and "search" in user_message.lower():
                mcp_result = await mcp_integration.call_mcp_tool("web_search", {"query": user_message})
                if mcp_result:
                    result += f"\n\n**Additional Context from MCP:**\n{mcp_result}"

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

    # Initialize tools and MCP
    await tools_manager.initialize_tools()
    if mcp_integration.enabled:
        mcp_tools = await mcp_integration.list_tools()
        logger.info(f"MCP integration enabled with {len(mcp_tools)} tools")

    # Start background tasks
    cleanup_history.start()

    # Update bot status
    try:
        status_text = "your messages"
        if tools_manager.tools_loaded:
            status_text += " + Tools"
        if mcp_integration.enabled:
            status_text += " + MCP"
        status_text += " | Powered by Daemonrat"

        await bot.change_presence(
            status=discord.Status.online,
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=status_text
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

                # Get AI response with enhanced capabilities
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

# Enhanced commands
@bot.command(name="ping")
async def ping(ctx):
    """Check if the bot is responsive"""
    latency = round(bot.latency * 1000)
    await ctx.send(f"🏓 Pong! Latency: {latency}ms")

@bot.command(name="status")
async def status(ctx):
    """Get bot status information"""
    embed = discord.Embed(
        title="Enhanced Bot Status",
        color=discord.Color.green(),
        timestamp=datetime.now()
    )

    embed.add_field(name="Guilds", value=len(bot.guilds), inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Model", value=config.MODEL_NAME, inline=True)
    embed.add_field(name="OpenWebUI Tools", value="✅ Enabled" if tools_manager.tools_loaded else "❌ Disabled", inline=True)
    embed.add_field(name="MCP Integration", value="✅ Enabled" if mcp_integration.enabled else "❌ Disabled", inline=True)
    embed.add_field(name="Available Tools", value=len(tools_manager.available_tools), inline=True)

    await ctx.send(embed=embed)

@bot.command(name="tools")
async def list_tools(ctx):
    """List available tools"""
    if not tools_manager.tools_loaded:
        await ctx.send("Tools not loaded. Attempting to initialize...")
        if not await tools_manager.initialize_tools():
            await ctx.send("❌ Failed to initialize tools.")
            return

    if tools_manager.available_tools:
        tools_list = []
        for tool in tools_manager.available_tools[:10]:  # Limit to first 10
            name = tool.get('name', 'Unknown')
            desc = tool.get('description', 'No description')[:50]
            tools_list.append(f"**{name}**: {desc}")

        embed = discord.Embed(
            title="Available OpenWebUI Tools",
            description="\n".join(tools_list),
            color=discord.Color.blue()
        )

        if len(tools_manager.available_tools) > 10:
            embed.add_field(name="Note", value=f"Showing 10 of {len(tools_manager.available_tools)} tools", inline=False)

        await ctx.send(embed=embed)
    else:
        await ctx.send("No tools available.")

@bot.command(name="mcp_tools")
async def list_mcp_tools(ctx):
    """List available MCP tools"""
    if not mcp_integration.enabled:
        await ctx.send("MCP integration is not enabled.")
        return

    mcp_tools = await mcp_integration.list_tools()
    if mcp_tools:
        tools_list = []
        for tool in mcp_tools[:10]:
            name = tool.get('name', 'Unknown')
            desc = tool.get('description', 'No description')[:50]
            tools_list.append(f"**{name}**: {desc}")

        embed = discord.Embed(
            title="Available MCP Tools",
            description="\n".join(tools_list),
            color=discord.Color.purple()
        )
        await ctx.send(embed=embed)
    else:
        await ctx.send("No MCP tools available.")

@bot.command(name="help_ai")
async def help_ai(ctx):
    """Show help information about the enhanced AI bot"""
    embed = discord.Embed(
        title="Enhanced OpenWebUI Discord Bot Help",
        description="I'm an AI assistant with advanced capabilities:",
        color=discord.Color.blue()
    )

    embed.add_field(
        name="💬 Chat with me",
        value="• Mention me: `@botname your question`\n• Send me a DM\n• I can process images you attach",
        inline=False
    )

    embed.add_field(
        name="🛠️ Enhanced Features",
        value="• OpenWebUI Tools integration\n• Model Context Protocol (MCP) support\n• Advanced function calling\n• Real-time tool loading",
        inline=False
    )

    embed.add_field(
        name="🤖 Commands",
        value="• `!ping` - Check responsiveness\n• `!status` - View detailed bot status\n• `!tools` - List OpenWebUI tools\n• `!mcp_tools` - List MCP tools",
        inline=False
    )

    await ctx.send(embed=embed)

def main():
    try:
        logger.info("Starting Enhanced Discord bot...")
        logger.info(f"OpenWebUI Tools: {'Enabled' if tools_manager else 'Disabled'}")
        logger.info(f"MCP Integration: {'Enabled' if mcp_integration.enabled else 'Disabled'}")

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