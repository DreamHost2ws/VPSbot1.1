# LightPlays_bot.py
import os
import asyncio
import logging
from dotenv import load_dotenv
import discord
from discord.ext import commands
import aiosqlite
import docker
from datetime import datetime

# Load .env
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
ADMIN_IDS = set(int(x) for x in (os.getenv("ADMIN_IDS") or "").split(",") if x.strip())
ADMIN_ROLE_ID = int(os.getenv("ADMIN_ROLE_ID") or 0)
DEFAULT_OS_IMAGE = os.getenv("DEFAULT_OS_IMAGE", "ubuntu:22.04")
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "bridge")
MAX_CONTAINERS = int(os.getenv("MAX_CONTAINERS", "100"))
MAX_VPS_PER_USER = int(os.getenv("MAX_VPS_PER_USER", "2"))

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vpsbot")

# Docker client (local Unix socket). For remote Docker, set environment DOCKER_HOST.
docker_client = docker.from_env()

# Discord bot
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

DB_PATH = "vps.db"

# Ensure DB exists
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS containers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            container_id TEXT UNIQUE,
            owner_id INTEGER,
            name TEXT,
            image TEXT,
            created_at TEXT,
            status TEXT
        )
        """)
        await db.commit()

@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (id: {bot.user.id})")
    await init_db()

def is_admin(user: discord.User):
    return user.id in ADMIN_IDS

# Helper: count user's active VPS
async def user_vps_count(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT COUNT(*) FROM containers WHERE owner_id=? AND status IN ('running','created')", (user_id,))
        row = await cur.fetchone()
        return row[0] if row else 0

# Command: create vps
@bot.command(name="createvps")
async def create_vps(ctx, name: str = None, image: str = None):
    owner = ctx.author
    if await user_vps_count(owner.id) >= MAX_VPS_PER_USER:
        await ctx.reply(f"You've reached your VPS limit ({MAX_VPS_PER_USER}). Delete one to create another.")
        return

    image = image or DEFAULT_OS_IMAGE
    name = (name or f"vps-{owner.id}-{int(datetime.utcnow().timestamp())}").lower()

    # Validate name
    if not name.isalnum() and "-" not in name:
        await ctx.reply("Invalid name. Use letters, numbers, and hyphens only.")
        return

    # Check global container count
    all_containers = docker_client.containers.list(all=True)
    if len(all_containers) >= MAX_CONTAINERS:
        await ctx.reply("Server is at max container capacity, try again later.")
        return

    await ctx.reply(f"Creating VPS `{name}` from image `{image}` — this may take a moment...")

    try:
        # Pull image if not present
        docker_client.images.pull(image)
        # Create container (adjust resources limits as needed)
        container = docker_client.containers.run(
            image,
            command="/bin/bash -c 'while true; do sleep 86400; done'",
            detach=True,
            name=name,
            network=DOCKER_NETWORK,
            tty=True,
            stdin_open=True,
            labels={"owner": str(owner.id)},
            host_config=docker_client.api.create_host_config(auto_remove=False)
        )
    except Exception as e:
        log.exception("Failed to create container")
        await ctx.reply(f"Failed to create container: {e}")
        return

    # Save metadata
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO containers (container_id, owner_id, name, image, created_at, status) VALUES (?, ?, ?, ?, ?, ?)",
                         (container.id, owner.id, name, image, datetime.utcnow().isoformat(), "running"))
        await db.commit()

    # Obtain IP/ports: note: for bridge network, containers have internal IP; to expose services, map ports
    # For now return container.id
    await ctx.reply(f"VPS `{name}` created. Container ID: `{container.id[:12]}`. Use `!listvps` to see your VPSes.")

# Command: list vps
@bot.command(name="listvps")
async def list_vps(ctx):
    owner = ctx.author
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT container_id, name, image, created_at, status FROM containers WHERE owner_id=?", (owner.id,))
        rows = await cur.fetchall()
    if not rows:
        await ctx.reply("You have no VPSes.")
        return
    msg = "Your VPSes:\n"
    for container_id, name, image, created_at, status in rows:
        short_id = container_id[:12]
        msg += f"- `{name}` ({short_id}) image={image} status={status} created={created_at}\n"
    await ctx.reply(msg)

# Command: stop vps
@bot.command(name="stopvps")
async def stop_vps(ctx, name_or_id: str):
    owner = ctx.author
    # Find container
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT container_id, name FROM containers WHERE owner_id=? AND (name=? OR container_id LIKE ?)", (owner.id, name_or_id, f"{name_or_id}%"))
        row = await cur.fetchone()
    if not row:
        await ctx.reply("No matching VPS found.")
        return
    container_id, name = row
    try:
        cont = docker_client.containers.get(container_id)
        cont.stop(timeout=10)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE containers SET status=? WHERE container_id=?", ("stopped", container_id))
            await db.commit()
    except Exception as e:
        await ctx.reply(f"Failed to stop: {e}")
        return
    await ctx.reply(f"Stopped VPS `{name}` ({container_id[:12]}).")

# Command: delete vps
@bot.command(name="deletevps")
async def delete_vps(ctx, name_or_id: str):
    owner = ctx.author
    # Find container
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT container_id, name FROM containers WHERE owner_id=? AND (name=? OR container_id LIKE ?)", (owner.id, name_or_id, f"{name_or_id}%"))
        row = await cur.fetchone()
    if not row:
        await ctx.reply("No matching VPS found.")
        return
    container_id, name = row
    try:
        cont = docker_client.containers.get(container_id)
        cont.remove(force=True)
    except docker.errors.NotFound:
        pass
    except Exception as e:
        await ctx.reply(f"Failed to remove: {e}")
        return
    # Remove metadata
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM containers WHERE container_id=?", (container_id,))
        await db.commit()
    await ctx.reply(f"Deleted VPS `{name}` ({container_id[:12]}).")

# Admin command example
@bot.command(name="purge_all")
async def purge_all(ctx):
    if not is_admin(ctx.author):
        await ctx.reply("You are not allowed to run this command.")
        return
    await ctx.reply("Purging all stopped containers and clearing DB entries...")
    # remove stopped containers with specific label
    for c in docker_client.containers.list(all=True, filters={"label": "owner"}):
        if c.status in ("exited","created","dead"):
            try:
                c.remove(force=True)
            except Exception:
                pass
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM containers WHERE status!='running'")
        await db.commit()
    await ctx.reply("Purge complete.")

# Run bot
if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN not set in environment.")
    bot.run(TOKEN)
