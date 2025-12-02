import discord
from discord.ext import commands
import asyncio
import subprocess
import json
from datetime import datetime
import shlex
import logging
import shutil
import os
from typing import Optional, List, Dict, Any
import threading
import time

# Load environment variables
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
MAIN_ADMIN_ID = int(os.getenv('MAIN_ADMIN_ID', '1210291131301101618'))
VPS_USER_ROLE_ID = int(os.getenv('VPS_USER_ROLE_ID', '1210291131301101618'))
DEFAULT_STORAGE_POOL = os.getenv('DEFAULT_STORAGE_POOL', 'default')
CPU_THRESHOLD = int(os.getenv('CPU_THRESHOLD', '90'))
RAM_THRESHOLD = int(os.getenv('RAM_THRESHOLD', '90'))
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '600'))  # 10 minutes for VPS monitoring

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BharatCloud_vps_bot')

# Check if lxc command is available
if not shutil.which("lxc"):
    logger.error("LXC command not found. Please ensure LXC is installed.")
    raise SystemExit("LXC command not found. Please ensure LXC is installed.")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Disable the default help command
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# CPU monitoring settings
cpu_monitor_active = True

# Helper function to truncate text to a specific length
def truncate_text(text, max_length=1024):
    """Truncate text to max_length characters"""
    if not text:
        return text
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

# Embed creation functions with black theme and BharatCloud branding
def create_embed(title, description="", color=0x1a1a1a):
    """Create a dark-themed embed with proper field length handling and BharatCloud branding"""
    embed = discord.Embed(
        title=truncate_text(f"⭐ BharatCloud - {title}", 256),
        description=truncate_text(description, 4096),
        color=color
    )

    embed.set_thumbnail(url="https://i.ibb.co/mVfMFfTR/e4c140ce1ab786fef61ac54974341089.png")
    embed.set_footer(text=f"BharatCloud VPS Manager • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    icon_url="https://i.ibb.co/mVfMFfTR/e4c140ce1ab786fef61ac54974341089.png")

    return embed

def add_field(embed, name, value, inline=False):
    """Add a field to an embed with proper truncation"""
    embed.add_field(
        name=truncate_text(f"▸ {name}", 256),
        value=truncate_text(value, 1024),
        inline=inline
    )
    return embed

def create_success_embed(title, description=""):
    return create_embed(title, description, color=0x00ff88)

def create_error_embed(title, description=""):
    return create_embed(title, description, color=0xff3366)

def create_info_embed(title, description=""):
    return create_embed(title, description, color=0x00ccff)

def create_warning_embed(title, description=""):
    return create_embed(title, description, color=0xffaa00)

# Data storage functions
def load_vps_data():
    try:
        with open('vps_data.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("vps_data.json not found or corrupted, initializing empty data")
        return {}

def load_admin_data():
    try:
        with open('admin_data.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("admin_data.json not found or corrupted, initializing with main admin")
        return {"admins": [str(MAIN_ADMIN_ID)]}

# Load all data at startup
vps_data = load_vps_data()
admin_data = load_admin_data()

# Save data function
def save_data():
    try:
        with open('vps_data.json', 'w') as f:
            json.dump(vps_data, f, indent=4)
        with open('admin_data.json', 'w') as f:
            json.dump(admin_data, f, indent=4)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

# Admin checks - Updated to not send message in predicate, more specific errors
def is_admin():
    async def predicate(ctx):
        user_id = str(ctx.author.id)
        if user_id == str(MAIN_ADMIN_ID) or user_id in admin_data.get("admins", []):
            return True
        # Custom error handling moved to on_command_error for better UX
        raise commands.CheckFailure(f"You need admin permissions to use this command. Contact BharatCloud support.")
    return commands.check(predicate)

def is_main_admin():
    async def predicate(ctx):
        if str(ctx.author.id) == str(MAIN_ADMIN_ID):
            return True
        raise commands.CheckFailure("Only the main admin can use this command.")
    return commands.check(predicate)

# Clean LXC command execution
async def execute_lxc(command, timeout=120):
    """Execute LXC command with timeout and error handling"""
    try:
        cmd = shlex.split(command)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        if proc.returncode != 0:
            error = stderr.decode().strip() if stderr else "Command failed with no error output"
            raise Exception(error)

        return stdout.decode().strip() if stdout else True
    except asyncio.TimeoutError:
        logger.error(f"LXC command timed out: {command}")
        raise Exception(f"Command timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"LXC Error: {command} - {str(e)}")
        raise

# Get or create VPS user role
async def get_or_create_vps_role(guild):
    """Get or create the VPS User role"""
    global VPS_USER_ROLE_ID
    
    if VPS_USER_ROLE_ID:
        role = guild.get_role(VPS_USER_ROLE_ID)
        if role:
            return role
    
    role = discord.utils.get(guild.roles, name="BharatCloud VPS User")
    if role:
        VPS_USER_ROLE_ID = role.id
        return role
    
    try:
        role = await guild.create_role(
            name="BharatCloud VPS User",
            color=discord.Color.dark_purple(),
            reason="BharatCloud VPS User role for bot management",
            permissions=discord.Permissions.none()
        )
        VPS_USER_ROLE_ID = role.id
        logger.info(f"Created BharatCloud VPS User role: {role.name} (ID: {role.id})")
        return role
    except Exception as e:
        logger.error(f"Failed to create BharatCloud VPS User role: {e}")
        return None

# Host CPU monitoring function
def get_cpu_usage():
    """Get current CPU usage percentage"""
    try:
        # Get CPU usage using top command
        result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
        output = result.stdout
        
        # Parse the output to get CPU usage
        for line in output.split('\n'):
            if '%Cpu(s):' in line:
                words = line.split()
                for i, word in enumerate(words):
                    if word == 'id,':
                        idle_str = words[i-1].rstrip(',')
                        try:
                            idle = float(idle_str)
                            usage = 100.0 - idle
                            return usage
                        except ValueError:
                            pass
                break
        return 0.0
    except Exception as e:
        logger.error(f"Error getting CPU usage: {e}")
        return 0.0

def cpu_monitor():
    """Monitor CPU usage and stop all VPS if threshold is exceeded"""
    global cpu_monitor_active
    
    while cpu_monitor_active:
        try:
            cpu_usage = get_cpu_usage()
            logger.info(f"Current CPU usage: {cpu_usage}%")
            
            if cpu_usage > CPU_THRESHOLD:
                logger.warning(f"CPU usage ({cpu_usage}%) exceeded threshold ({CPU_THRESHOLD}%). Stopping all VPS.")
                
                # Execute lxc stop --all --force
                try:
                    subprocess.run(['lxc', 'stop', '--all', '--force'], check=True)
                    logger.info("All VPS stopped due to high CPU usage")
                    
                    # Update all VPS status in database
                    for user_id, vps_list in vps_data.items():
                        for vps in vps_list:
                            if vps.get('status') == 'running':
                                vps['status'] = 'stopped'
                    save_data()
                except Exception as e:
                    logger.error(f"Error stopping all VPS: {e}")
            
            time.sleep(60)  # Check host every 60 seconds
        except Exception as e:
            logger.error(f"Error in CPU monitor: {e}")
            time.sleep(60)

# Start CPU monitoring in a separate thread
cpu_thread = threading.Thread(target=cpu_monitor, daemon=True)
cpu_thread.start()

# Helper functions for container stats
async def get_container_status(container_name):
    """Get the status of the LXC container"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "info", container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode()
        for line in output.splitlines():
            if line.startswith("Status: "):
                return line.split(": ", 1)[1].strip()
        return "Unknown"
    except Exception:
        return "Unknown"

async def get_container_cpu(container_name):
    """Get CPU usage inside the container as string"""
    usage = await get_container_cpu_pct(container_name)
    return f"{usage:.1f}%"

async def get_container_cpu_pct(container_name):
    """Get CPU usage percentage inside the container as float"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "exec", container_name, "--", "top", "-bn1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode()
        for line in output.splitlines():
            if '%Cpu(s):' in line:
                words = line.split()
                for i, word in enumerate(words):
                    if word == 'id,':
                        idle_str = words[i-1].rstrip(',')
                        try:
                            idle = float(idle_str)
                            usage = 100.0 - idle
                            return usage
                        except ValueError:
                            pass
                break
        return 0.0
    except Exception as e:
        logger.error(f"Error getting CPU for {container_name}: {e}")
        return 0.0

async def get_container_memory(container_name):
    """Get memory usage inside the container"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "exec", container_name, "--", "free", "-m",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode().splitlines()
        if len(lines) > 1:
            parts = lines[1].split()
            total = int(parts[1])
            used = int(parts[2])
            usage_pct = (used / total * 100) if total > 0 else 0
            return f"{used}/{total} MB ({usage_pct:.1f}%)"
        return "Unknown"
    except Exception:
        return "Unknown"

async def get_container_ram_pct(container_name):
    """Get RAM usage percentage inside the container as float"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "exec", container_name, "--", "free", "-m",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode().splitlines()
        if len(lines) > 1:
            parts = lines[1].split()
            total = int(parts[1])
            used = int(parts[2])
            usage_pct = (used / total * 100) if total > 0 else 0
            return usage_pct
        return 0.0
    except Exception as e:
        logger.error(f"Error getting RAM for {container_name}: {e}")
        return 0.0

async def get_container_disk(container_name):
    """Get disk usage inside the container"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "exec", container_name, "--", "df", "-h", "/",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode().splitlines()
        for line in lines:
            if '/dev/' in line and ' /' in line:
                parts = line.split()
                if len(parts) >= 5:
                    used = parts[2]
                    size = parts[1]
                    perc = parts[4]
                    return f"{used}/{size} ({perc})"
        return "Unknown"
    except Exception:
        return "Unknown"

def get_uptime():
    """Get host uptime"""
    try:
        result = subprocess.run(['uptime'], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "Unknown"

# VPS monitoring task
async def vps_monitor():
    """Monitor each VPS for high CPU/RAM usage every 10 minutes"""
    while True:
        try:
            for user_id, vps_list in vps_data.items():
                for vps in vps_list:
                    if vps.get('status') == 'running' and not vps.get('suspended', False):
                        container = vps['container_name']
                        cpu = await get_container_cpu_pct(container)
                        ram = await get_container_ram_pct(container)
                        if cpu > CPU_THRESHOLD or ram > RAM_THRESHOLD:
                            reason = f"High resource usage: CPU {cpu:.1f}%, RAM {ram:.1f}% (threshold: {CPU_THRESHOLD}% CPU / {RAM_THRESHOLD}% RAM)"
                            logger.warning(f"Suspending {container}: {reason}")
                            try:
                                await execute_lxc(f"lxc stop {container}")
                                vps['status'] = 'suspended'
                                vps['suspended'] = True
                                if 'suspension_history' not in vps:
                                    vps['suspension_history'] = []
                                vps['suspension_history'].append({
                                    'time': datetime.now().isoformat(),
                                    'reason': reason,
                                    'by': 'BharatCloud Auto-System'
                                })
                                save_data()
                                # DM owner
                                try:
                                    owner = await bot.fetch_user(int(user_id))
                                    embed = create_warning_embed("🚨 VPS Auto-Suspended", f"Your VPS `{container}` has been automatically suspended due to high resource usage.\n\n**Reason:** {reason}\n\nContact BharatCloud admin to unsuspend and address the issue.")
                                    await owner.send(embed=embed)
                                except Exception as dm_e:
                                    logger.error(f"Failed to DM owner {user_id}: {dm_e}")
                            except Exception as e:
                                logger.error(f"Failed to suspend {container}: {e}")
            await asyncio.sleep(CHECK_INTERVAL)
        except Exception as e:
            logger.error(f"VPS monitor error: {e}")
            await asyncio.sleep(60)

# Bot events
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="BharatCloud VPS Manager"))
    bot.loop.create_task(vps_monitor())
    logger.info("BharatCloud Bot is ready! VPS monitoring started.")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(embed=create_error_embed("Missing Argument", "Please check command usage with `!help`."))
    elif isinstance(error, commands.BadArgument):
        await ctx.send(embed=create_error_embed("Invalid Argument", "Please check your input and try again."))
    elif isinstance(error, commands.CheckFailure):
        # More user-friendly error without generic message
        error_msg = str(error) if str(error) != "Admin required" else "You need admin permissions for this command. Contact BharatCloud support."
        await ctx.send(embed=create_error_embed("Access Denied", error_msg))
    elif isinstance(error, discord.NotFound):
        # Handle 404 Not Found errors
        await ctx.send(embed=create_error_embed("Error", "The requested resource was not found. Please try again."))
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(embed=create_error_embed("System Error", "An unexpected error occurred. BharatCloud support has been notified."))

# Bot commands
@bot.command(name='ping')
async def ping(ctx):
    """Check bot latency"""
    latency = round(bot.latency * 1000)
    embed = create_success_embed("Pong!", f"BharatCloud Bot latency: {latency}ms")
    await ctx.send(embed=embed)

@bot.command(name='uptime')
async def uptime(ctx):
    """Show host uptime"""
    up = get_uptime()
    embed = create_info_embed("Host Uptime", up)
    await ctx.send(embed=embed)

@bot.command(name='myvps')
async def my_vps(ctx):
    """List your VPS"""
    user_id = str(ctx.author.id)
    vps_list = vps_data.get(user_id, [])
    if not vps_list:
        await ctx.send(embed=create_embed("No VPS Found", "You don't have any BharatCloud VPS. Contact an admin to create one.", 0xff3366))
        return
    embed = create_info_embed("My BharatCloud VPS", "")
    text = []
    for i, vps in enumerate(vps_list):
        status = vps.get('status', 'unknown').upper()
        if vps.get('suspended', False):
            status += " (SUSPENDED)"
        config = vps.get('config', 'Custom')
        text.append(f"**VPS {i+1}:** `{vps['container_name']}` - {status} - {config}")
    add_field(embed, "Your VPS", "\n".join(text), False)
    add_field(embed, "Actions", "Use `!manage` to start/stop/reinstall", False)
    await ctx.send(embed=embed)

@bot.command(name='lxc-list')
@is_admin()
async def lxc_list(ctx):
    """List all LXC containers"""
    try:
        result = await execute_lxc("lxc list")
        embed = create_info_embed("BharatCloud LXC Containers List", result)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(embed=create_error_embed("Error", str(e)))

@bot.command(name='create')
@is_admin()
async def create_vps(ctx, ram: int, cpu: int, disk: int, user: discord.Member):
    """Create a custom VPS for a user (Admin only) - !create <ram_gb> <cpu_cores> <disk_gb> <user>"""
    if ram <= 0 or cpu <= 0 or disk <= 0:
        await ctx.send(embed=create_error_embed("Invalid Specs", "RAM, CPU, and Disk must be positive integers."))
        return

    user_id = str(user.id)
    if user_id not in vps_data:
        vps_data[user_id] = []

    vps_count = len(vps_data[user_id]) + 1
    container_name = f"BharatCloud-vps-{user_id}-{vps_count}"
    ram_mb = ram * 1024

    await ctx.send(embed=create_info_embed("Creating BharatCloud VPS", f"Deploying VPS for {user.mention}..."))

    try:
        # Fixed: Use init for config before start
        await execute_lxc(f"lxc init ubuntu:22.04 {container_name} --storage {DEFAULT_STORAGE_POOL}")
        await execute_lxc(f"lxc config set {container_name} limits.memory {ram_mb}MB")
        await execute_lxc(f"lxc config set {container_name} limits.cpu {cpu}")
        
        # Always resize the disk to specified size
        await execute_lxc(f"lxc config device set {container_name} root size {disk}GB")
        # Start to apply changes
        await execute_lxc(f"lxc start {container_name}")

        config_str = f"{ram}GB RAM / {cpu} CPU / {disk}GB Disk"
        vps_info = {
            "container_name": container_name,
            "ram": f"{ram}GB",
            "cpu": str(cpu),
            "storage": f"{disk}GB",
            "config": config_str,
            "status": "running",
            "suspended": False,
            "suspension_history": [],
            "created_at": datetime.now().isoformat(),
            "shared_with": []
        }
        vps_data[user_id].append(vps_info)
        save_data()

        # Get or create VPS role and assign to user
        if ctx.guild:
            vps_role = await get_or_create_vps_role(ctx.guild)
            if vps_role:
                try:
                    await user.add_roles(vps_role, reason="BharatCloud VPS ownership granted")
                except discord.Forbidden:
                    logger.warning(f"Failed to assign BharatCloud VPS role to {user.name}")

        # Create success embed for channel
        embed = create_success_embed("BharatCloud VPS Created Successfully")
        add_field(embed, "Owner", user.mention, True)
        add_field(embed, "VPS ID", f"#{vps_count}", True)
        add_field(embed, "Container", f"`{container_name}`", True)
        add_field(embed, "Resources", f"**RAM:** {ram}GB\n**CPU:** {cpu} Cores\n**Storage:** {disk}GB", False)
        await ctx.send(embed=embed)

        # Send comprehensive DM to user
        try:
            dm_embed = create_success_embed("BharatCloud VPS Created!", f"Your VPS has been successfully deployed by an admin!")
            add_field(dm_embed, "VPS Details", f"**VPS ID:** #{vps_count}\n**Container Name:** `{container_name}`\n**Configuration:** {config_str}\n**Status:** Running\n**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", False)
            add_field(dm_embed, "Management", "• Use `!manage` to start/stop/reinstall your BharatCloud VPS\n• Use `!manage` → SSH for terminal access\n• Contact BharatCloud admin for upgrades or issues", False)
            add_field(dm_embed, "Important Notes", "• Full root access via SSH\n• Ubuntu 22.04 pre-installed\n• Back up your data regularly with BharatCloud tools", False)
            await user.send(embed=dm_embed)
        except discord.Forbidden:
            await ctx.send(embed=create_info_embed("Notification Failed", f"Couldn't send DM to {user.mention}. Please ensure DMs are enabled."))

    except Exception as e:
        await ctx.send(embed=create_error_embed("Creation Failed", f"Error: {str(e)}"))

class ManageView(discord.ui.View):
    def __init__(self, user_id, vps_list, is_shared=False, owner_id=None, is_admin=False):
        super().__init__(timeout=300)
        self.user_id = user_id
        self.vps_list = vps_list
        self.selected_index = None
        self.is_shared = is_shared
        self.owner_id = owner_id or user_id
        self.is_admin = is_admin

        if len(vps_list) > 1:
            options = [
                discord.SelectOption(
                    label=f"BharatCloud VPS {i+1} ({v.get('config', 'Custom')})",
                    description=f"Status: {v.get('status', 'unknown')}",
                    value=str(i)
                ) for i, v in enumerate(vps_list)
            ]
            self.select = discord.ui.Select(placeholder="Select a BharatCloud VPS to manage", options=options)
            self.select.callback = self.select_vps
            self.add_item(self.select)
            self.initial_embed = create_embed("BharatCloud VPS Management", "Select a VPS from the dropdown menu below.", 0x1a1a1a)
            add_field(self.initial_embed, "Available VPS", "\n".join([f"**VPS {i+1}:** `{v['container_name']}` - Status: `{v.get('status', 'unknown').upper()}`" for i, v in enumerate(vps_list)]), False)
        else:
            self.selected_index = 0
            self.initial_embed = None
            self.add_action_buttons()

    async def get_initial_embed(self):
        if self.initial_embed is not None:
            return self.initial_embed
        self.initial_embed = await self.create_vps_embed(self.selected_index)
        return self.initial_embed

    async def create_vps_embed(self, index):
        vps = self.vps_list[index]
        status = vps.get('status', 'unknown')
        suspended = vps.get('suspended', False)
        status_color = 0x00ff88 if status == 'running' and not suspended else 0xffaa00 if suspended else 0xff3366

        # Fetch live stats
        container_name = vps['container_name']
        lxc_status = await get_container_status(container_name)
        cpu_usage = await get_container_cpu(container_name)
        memory_usage = await get_container_memory(container_name)
        disk_usage = await get_container_disk(container_name)

        status_text = f"{status.upper()}"
        if suspended:
            status_text += " (SUSPENDED)"

        owner_text = ""
        if self.is_admin and self.owner_id != self.user_id:
            try:
                owner_user = bot.get_user(int(self.owner_id))
                owner_text = f"\n**Owner:** {owner_user.mention}"
            except:
                owner_text = f"\n**Owner ID:** {self.owner_id}"

        embed = create_embed(
            f"BharatCloud VPS Management - VPS {index + 1}",
            f"Managing container: `{container_name}`{owner_text}",
            status_color
        )

        resource_info = f"**Configuration:** {vps.get('config', 'Custom')}\n"
        resource_info += f"**Status:** `{status_text}`\n"
        resource_info += f"**RAM:** {vps['ram']}\n"
        resource_info += f"**CPU:** {vps['cpu']} Cores\n"
        resource_info += f"**Storage:** {vps['storage']}"

        add_field(embed, "📊 Allocated Resources", resource_info, False)

        if suspended:
            add_field(embed, "⚠️ Suspended", "This BharatCloud VPS is suspended. Contact an admin to unsuspend.", False)

        live_stats = f"**CPU Usage:** {cpu_usage}\n**Memory:** {memory_usage}\n**Disk:** {disk_usage}"
        add_field(embed, "📈 Live Usage", live_stats, False)

        add_field(embed, "🎮 Controls", "Use the buttons below to manage your BharatCloud VPS", False)

        return embed

    def add_action_buttons(self):
        if not self.is_shared and not self.is_admin:
            reinstall_button = discord.ui.Button(label="🔄 Reinstall", style=discord.ButtonStyle.danger)
            reinstall_button.callback = lambda inter: self.action_callback(inter, 'reinstall')
            self.add_item(reinstall_button)

        start_button = discord.ui.Button(label="▶ Start", style=discord.ButtonStyle.success)
        start_button.callback = lambda inter: self.action_callback(inter, 'start')
        stop_button = discord.ui.Button(label="⏸ Stop", style=discord.ButtonStyle.secondary)
        stop_button.callback = lambda inter: self.action_callback(inter, 'stop')
        ssh_button = discord.ui.Button(label="🔑 SSH", style=discord.ButtonStyle.primary)
        ssh_button.callback = lambda inter: self.action_callback(inter, 'tmate')
        stats_button = discord.ui.Button(label="📊 Stats", style=discord.ButtonStyle.secondary)
        stats_button.callback = lambda inter: self.action_callback(inter, 'stats')

        self.add_item(start_button)
        self.add_item(stop_button)
        self.add_item(ssh_button)
        self.add_item(stats_button)

    async def select_vps(self, interaction: discord.Interaction):
        if str(interaction.user.id) != self.user_id and not self.is_admin:
            await interaction.response.send_message(embed=create_error_embed("Access Denied", "This is not your BharatCloud VPS!"), ephemeral=True)
            return
        self.selected_index = int(self.select.values[0])
        new_embed = await self.create_vps_embed(self.selected_index)
        self.clear_items()
        self.add_action_buttons()
        await interaction.response.edit_message(embed=new_embed, view=self)

    async def action_callback(self, interaction: discord.Interaction, action: str):
        if str(interaction.user.id) != self.user_id and not self.is_admin:
            await interaction.response.send_message(embed=create_error_embed("Access Denied", "This is not your BharatCloud VPS!"), ephemeral=True)
            return

        if self.is_shared:
            vps = vps_data[self.owner_id][self.selected_index]
        else:
            vps = self.vps_list[self.selected_index]
        
        suspended = vps.get('suspended', False)
        if suspended and not self.is_admin and action != 'stats':
            await interaction.response.send_message(embed=create_error_embed("Access Denied", "This BharatCloud VPS is suspended. Contact an admin to unsuspend."), ephemeral=True)
            return
        
        container_name = vps["container_name"]

        if action == 'stats':
            status = await get_container_status(container_name)
            cpu_usage = await get_container_cpu(container_name)
            memory_usage = await get_container_memory(container_name)
            disk_usage = await get_container_disk(container_name)
            stats_embed = create_info_embed("📈 BharatCloud Live Statistics", f"Real-time stats for `{container_name}`")
            add_field(stats_embed, "Status", f"`{status.upper()}`", True)
            add_field(stats_embed, "CPU", cpu_usage, True)
            add_field(stats_embed, "Memory", memory_usage, True)
            add_field(stats_embed, "Disk", disk_usage, True)
            await interaction.response.send_message(embed=stats_embed, ephemeral=True)
            return

        if action == 'reinstall':
            if self.is_shared or self.is_admin:
                await interaction.response.send_message(embed=create_error_embed("Access Denied", "Only the BharatCloud VPS owner can reinstall!"), ephemeral=True)
                return
            if suspended:
                await interaction.response.send_message(embed=create_error_embed("Cannot Reinstall", "Unsuspend the BharatCloud VPS first."), ephemeral=True)
                return

            confirm_embed = create_warning_embed("BharatCloud Reinstall Warning",
                f"⚠️ **WARNING:** This will erase all data on VPS `{container_name}` and reinstall Ubuntu 22.04.\n\n"
                f"This action cannot be undone. Continue?")

            class ConfirmView(discord.ui.View):
                def __init__(self, parent_view, container_name, vps, owner_id, selected_index):
                    super().__init__(timeout=60)
                    self.parent_view = parent_view
                    self.container_name = container_name
                    self.vps = vps
                    self.owner_id = owner_id
                    self.selected_index = selected_index

                @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
                async def confirm(self, interaction: discord.Interaction, item: discord.ui.Button):
                    await interaction.response.defer(ephemeral=True)
                    try:
                        # Force delete the container first
                        await interaction.followup.send(embed=create_info_embed("Deleting Container", f"Forcefully removing container `{self.container_name}`..."), ephemeral=True)
                        await execute_lxc(f"lxc delete {self.container_name} --force")

                        # Recreate with original specifications - Fixed init + start
                        await interaction.followup.send(embed=create_info_embed("Recreating Container", f"Creating new BharatCloud container `{self.container_name}`..."), ephemeral=True)
                        original_ram = self.vps["ram"]
                        original_cpu = self.vps["cpu"]
                        original_storage = self.vps["storage"]
                        ram_gb = int(original_ram.replace("GB", ""))
                        ram_mb = ram_gb * 1024
                        storage_gb = int(original_storage.replace("GB", ""))

                        await execute_lxc(f"lxc init ubuntu:22.04 {self.container_name} --storage {DEFAULT_STORAGE_POOL}")
                        await execute_lxc(f"lxc config set {self.container_name} limits.memory {ram_mb}MB")
                        await execute_lxc(f"lxc config set {self.container_name} limits.cpu {original_cpu}")
                        await execute_lxc(f"lxc config device set {self.container_name} root size {storage_gb}GB")
                        await execute_lxc(f"lxc start {self.container_name}")

                        self.vps["status"] = "running"
                        self.vps["suspended"] = False
                        self.vps["created_at"] = datetime.now().isoformat()
                        config_str = f"{ram_gb}GB RAM / {original_cpu} CPU / {storage_gb}GB Disk"
                        self.vps["config"] = config_str
                        save_data()
                        await interaction.followup.send(embed=create_success_embed("Reinstall Complete", f"BharatCloud VPS `{self.container_name}` has been successfully reinstalled!"), ephemeral=True)

                        # Edit the original message if possible, but since ephemeral, send updated embed as followup
                        new_embed = await self.parent_view.create_vps_embed(self.parent_view.selected_index)
                        await interaction.followup.send(embed=new_embed, ephemeral=True)

                    except Exception as e:
                        await interaction.followup.send(embed=create_error_embed("Reinstall Failed", f"Error: {str(e)}"), ephemeral=True)

                @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
                async def cancel(self, interaction: discord.Interaction, item: discord.ui.Button):
                    new_embed = await self.parent_view.create_vps_embed(self.parent_view.selected_index)
                    await interaction.response.edit_message(embed=new_embed, view=self.parent_view)

            await interaction.response.send_message(embed=confirm_embed, view=ConfirmView(self, container_name, vps, self.owner_id, self.selected_index), ephemeral=True)

        elif action == 'start':
            await interaction.response.defer(ephemeral=True)
            if suspended:
                vps['suspended'] = False
                save_data()
            try:
                await execute_lxc(f"lxc start {container_name}")
                vps["status"] = "running"
                save_data()
                await interaction.followup.send(embed=create_success_embed("VPS Started", f"BharatCloud VPS `{container_name}` is now running!"), ephemeral=True)
                new_embed = await self.create_vps_embed(self.selected_index)
                await interaction.message.edit(embed=new_embed, view=self)
            except Exception as e:
                await interaction.followup.send(embed=create_error_embed("Start Failed", str(e)), ephemeral=True)

        elif action == 'stop':
            await interaction.response.defer(ephemeral=True)
            if suspended:
                vps['suspended'] = False
                save_data()
            try:
                await execute_lxc(f"lxc stop {container_name}", timeout=120)
                vps["status"] = "stopped"
                save_data()
                await interaction.followup.send(embed=create_success_embed("VPS Stopped", f"BharatCloud VPS `{container_name}` has been stopped!"), ephemeral=True)
                new_embed = await self.create_vps_embed(self.selected_index)
                await interaction.message.edit(embed=new_embed, view=self)
            except Exception as e:
                await interaction.followup.send(embed=create_error_embed("Stop Failed", str(e)), ephemeral=True)

        elif action == 'tmate':
            if suspended:
                await interaction.response.send_message(embed=create_error_embed("Access Denied", "Cannot access suspended BharatCloud VPS."), ephemeral=True)
                return
            await interaction.response.send_message(embed=create_info_embed("SSH Access", "Generating BharatCloud SSH connection..."), ephemeral=True)

            try:
                # Check if tmate exists
                check_proc = await asyncio.create_subprocess_exec(
                    "lxc", "exec", container_name, "--", "which", "tmate",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await check_proc.communicate()

                if check_proc.returncode != 0:
                    await interaction.followup.send(embed=create_info_embed("Installing SSH", "Installing tmate..."), ephemeral=True)
                    await execute_lxc(f"lxc exec {container_name} -- sudo apt-get update -y")
                    await execute_lxc(f"lxc exec {container_name} -- sudo apt-get install tmate -y")
                    await interaction.followup.send(embed=create_success_embed("Installed", "BharatCloud SSH service installed!"), ephemeral=True)

                # Start tmate with unique session name using timestamp
                session_name = f"BharatCloud-session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                await execute_lxc(f"lxc exec {container_name} -- tmate -S /tmp/{session_name}.sock new-session -d")
                await asyncio.sleep(3)

                # Get SSH link
                ssh_proc = await asyncio.create_subprocess_exec(
                    "lxc", "exec", container_name, "--", "tmate", "-S", f"/tmp/{session_name}.sock", "display", "-p", "#{tmate_ssh}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await ssh_proc.communicate()
                ssh_url = stdout.decode().strip() if stdout else None

                if ssh_url:
                    try:
                        ssh_embed = create_embed("🔑 BharatCloud SSH Access", f"SSH connection for VPS `{container_name}`:", 0x00ff88)
                        add_field(ssh_embed, "Command", f"```{ssh_url}```", False)
                        add_field(ssh_embed, "⚠️ Security", "This link is temporary. Do not share it.", False)
                        add_field(ssh_embed, "📝 Session", f"Session ID: {session_name}", False)
                        await interaction.user.send(embed=ssh_embed)
                        await interaction.followup.send(embed=create_success_embed("SSH Sent", f"Check your DMs for BharatCloud SSH link! Session: {session_name}"), ephemeral=True)
                    except discord.Forbidden:
                        await interaction.followup.send(embed=create_error_embed("DM Failed", "Enable DMs to receive BharatCloud SSH link!"), ephemeral=True)
                else:
                    error_msg = stderr.decode().strip() if stderr else "Unknown error"
                    await interaction.followup.send(embed=create_error_embed("SSH Failed", error_msg), ephemeral=True)
            except Exception as e:
                await interaction.followup.send(embed=create_error_embed("SSH Error", str(e)), ephemeral=True)

@bot.command(name='manage')
async def manage_vps(ctx, user: discord.Member = None):
    """Manage your BharatCloud VPS or another user's VPS (Admin only)"""
    # Check if user is trying to manage someone else's VPS
    if user:
        # Only admins can manage other users' VPS
        user_id_check = str(ctx.author.id)
        if user_id_check != str(MAIN_ADMIN_ID) and user_id_check not in admin_data.get("admins", []):
            await ctx.send(embed=create_error_embed("Access Denied", "Only BharatCloud admins can manage other users' VPS."))
            return
        
        user_id = str(user.id)
        vps_list = vps_data.get(user_id, [])
        if not vps_list:
            await ctx.send(embed=create_error_embed("No VPS Found", f"{user.mention} doesn't have any BharatCloud VPS."))
            return
        
        # Create admin view for managing another user's VPS
        view = ManageView(str(ctx.author.id), vps_list, is_admin=True, owner_id=user_id)
        await ctx.send(embed=create_info_embed(f"Managing {user.name}'s BharatCloud VPS", f"Managing VPS for {user.mention}"), view=view)
    else:
        # User managing their own VPS
        user_id = str(ctx.author.id)
        vps_list = vps_data.get(user_id, [])
        if not vps_list:
            embed = create_embed("No VPS Found", "You don't have any BharatCloud VPS. Contact an admin to create one.", 0xff3366)
            add_field(embed, "Quick Actions", "• `!manage` - Manage VPS\n• Contact BharatCloud admin for VPS creation", False)
            await ctx.send(embed=embed)
            return
        view = ManageView(user_id, vps_list)
        embed = await view.get_initial_embed()
        await ctx.send(embed=embed, view=view)

@bot.command(name='list-all')
@is_admin()
async def list_all_vps(ctx):
    """List all BharatCloud VPS and user information (Admin only)"""
    total_vps = 0
    total_users = len(vps_data)
    running_vps = 0
    stopped_vps = 0
    suspended_vps = 0
    
    vps_info = []
    user_summary = []
    
    for user_id, vps_list in vps_data.items():
        try:
            user = await bot.fetch_user(int(user_id))
            user_vps_count = len(vps_list)
            user_running = sum(1 for vps in vps_list if vps.get('status') == 'running' and not vps.get('suspended', False))
            user_stopped = sum(1 for vps in vps_list if vps.get('status') == 'stopped')
            user_suspended = sum(1 for vps in vps_list if vps.get('suspended', True))
            
            total_vps += user_vps_count
            running_vps += user_running
            stopped_vps += user_stopped
            suspended_vps += user_suspended
            
            # User summary
            user_summary.append(f"**{user.name}** ({user.mention}) - {user_vps_count} BharatCloud VPS ({user_running} running, {user_suspended} suspended)")
            
            # Individual VPS details
            for i, vps in enumerate(vps_list):
                status_emoji = "🟢" if vps.get('status') == 'running' and not vps.get('suspended', False) else "🟡" if vps.get('suspended', False) else "🔴"
                status_text = vps.get('status', 'unknown').upper()
                if vps.get('suspended', False):
                    status_text += " (SUSPENDED)"
                vps_info.append(f"{status_emoji} **{user.name}** - VPS {i+1}: `{vps['container_name']}` - {vps.get('config', 'Custom')} - {status_text}")
                
        except discord.NotFound:
            vps_info.append(f"❓ Unknown User ({user_id}) - {len(vps_list)} BharatCloud VPS")
    
    # Create multiple embeds if needed to avoid character limit
    embeds = []
    
    # First embed with overview
    embed = create_embed("All BharatCloud VPS Information", "Complete overview of all BharatCloud VPS deployments and user statistics", 0x1a1a1a)
    add_field(embed, "System Overview", f"**Total Users:** {total_users}\n**Total VPS:** {total_vps}\n**Running:** {running_vps}\n**Stopped:** {stopped_vps}\n**Suspended:** {suspended_vps}", False)
    embeds.append(embed)
    
    # User summary embed
    if user_summary:
        embed = create_embed("BharatCloud User Summary", f"Summary of all users and their BharatCloud VPS", 0x1a1a1a)
        # Split user summary into chunks to avoid character limit
        for i in range(0, len(user_summary), 10):
            chunk = user_summary[i:i+10]
            summary_text = "\n".join(chunk)
            if i == 0:
                add_field(embed, "Users", summary_text, False)
            else:
                add_field(embed, f"Users (continued {i+1}-{min(i+10, len(user_summary))})", summary_text, False)
        embeds.append(embed)
    
    # VPS details embeds
    if vps_info:
        # Split VPS info into chunks to avoid character limit
        for i in range(0, len(vps_info), 15):
            chunk = vps_info[i:i+15]
            embed = create_embed(f"BharatCloud VPS Details ({i+1}-{min(i+15, len(vps_info))})", "List of all BharatCloud VPS deployments", 0x1a1a1a)
            add_field(embed, "VPS List", "\n".join(chunk), False)
            embeds.append(embed)
    
    # Send all embeds
    for embed in embeds:
        await ctx.send(embed=embed)

@bot.command(name='manage-shared')
async def manage_shared_vps(ctx, owner: discord.Member, vps_number: int):
    """Manage a shared BharatCloud VPS"""
    owner_id = str(owner.id)
    user_id = str(ctx.author.id)
    if owner_id not in vps_data or vps_number < 1 or vps_number > len(vps_data[owner_id]):
        await ctx.send(embed=create_error_embed("Invalid VPS", "Invalid VPS number or owner doesn't have a BharatCloud VPS."))
        return
    vps = vps_data[owner_id][vps_number - 1]
    if user_id not in vps.get("shared_with", []):
        await ctx.send(embed=create_error_embed("Access Denied", "You do not have access to this BharatCloud VPS."))
        return
    view = ManageView(user_id, [vps], is_shared=True, owner_id=owner_id)
    embed = await view.get_initial_embed()
    await ctx.send(embed=embed, view=view)

@bot.command(name='share-user')
async def share_user(ctx, shared_user: discord.Member, vps_number: int):
    """Share BharatCloud VPS access with another user"""
    user_id = str(ctx.author.id)
    shared_user_id = str(shared_user.id)
    if user_id not in vps_data or vps_number < 1 or vps_number > len(vps_data[user_id]):
        await ctx.send(embed=create_error_embed("Invalid VPS", "Invalid VPS number or you don't have a BharatCloud VPS."))
        return
    vps = vps_data[user_id][vps_number - 1]

    if "shared_with" not in vps:
        vps["shared_with"] = []

    if shared_user_id in vps["shared_with"]:
        await ctx.send(embed=create_error_embed("Already Shared", f"{shared_user.mention} already has access to this BharatCloud VPS!"))
        return
    vps["shared_with"].append(shared_user_id)
    save_data()
    await ctx.send(embed=create_success_embed("VPS Shared", f"BharatCloud VPS #{vps_number} shared with {shared_user.mention}!"))
    try:
        await shared_user.send(embed=create_embed("BharatCloud VPS Access Granted", f"You have access to VPS #{vps_number} from {ctx.author.mention}. Use `!manage-shared {ctx.author.mention} {vps_number}`", 0x00ff88))
    except discord.Forbidden:
        await ctx.send(embed=create_info_embed("Notification Failed", f"Could not DM {shared_user.mention}"))

@bot.command(name='share-ruser')
async def revoke_share(ctx, shared_user: discord.Member, vps_number: int):
    """Revoke shared BharatCloud VPS access"""
    user_id = str(ctx.author.id)
    shared_user_id = str(shared_user.id)
    if user_id not in vps_data or vps_number < 1 or vps_number > len(vps_data[user_id]):
        await ctx.send(embed=create_error_embed("Invalid VPS", "Invalid VPS number or you don't have a BharatCloud VPS."))
        return
    vps = vps_data[user_id][vps_number - 1]

    if "shared_with" not in vps:
        vps["shared_with"] = []

    if shared_user_id not in vps["shared_with"]:
        await ctx.send(embed=create_error_embed("Not Shared", f"{shared_user.mention} doesn't have access to this BharatCloud VPS!"))
        return
    vps["shared_with"].remove(shared_user_id)
    save_data()
    await ctx.send(embed=create_success_embed("Access Revoked", f"Access to BharatCloud VPS #{vps_number} revoked from {shared_user.mention}!"))
    try:
        await shared_user.send(embed=create_embed("BharatCloud VPS Access Revoked", f"Your access to VPS #{vps_number} by {ctx.author.mention} has been revoked.", 0xff3366))
    except discord.Forbidden:
        await ctx.send(embed=create_info_embed("Notification Failed", f"Could not DM {shared_user.mention}"))

@bot.command(name='delete-vps')
@is_admin()
async def delete_vps(ctx, user: discord.Member, vps_number: int, *, reason: str = "No reason"):
    """Delete a user's BharatCloud VPS (Admin only)"""
    user_id = str(user.id)
    if user_id not in vps_data or vps_number < 1 or vps_number > len(vps_data[user_id]):
        await ctx.send(embed=create_error_embed("Invalid VPS", "Invalid VPS number or user doesn't have a BharatCloud VPS."))
        return
    vps = vps_data[user_id][vps_number - 1]
    container_name = vps["container_name"]

    await ctx.send(embed=create_info_embed("Deleting BharatCloud VPS", f"Removing VPS #{vps_number}..."))

    try:
        await execute_lxc(f"lxc delete {container_name} --force")
        del vps_data[user_id][vps_number - 1]
        if not vps_data[user_id]:
            del vps_data[user_id]
            # Remove VPS role if user has no more VPS
            if ctx.guild:
                vps_role = await get_or_create_vps_role(ctx.guild)
                if vps_role and vps_role in user.roles:
                    try:
                        await user.remove_roles(vps_role, reason="No BharatCloud VPS ownership")
                    except discord.Forbidden:
                        logger.warning(f"Failed to remove BharatCloud VPS role from {user.name}")
        save_data()

        embed = create_success_embed("BharatCloud VPS Deleted Successfully")
        add_field(embed, "Owner", user.mention, True)
        add_field(embed, "VPS ID", f"#{vps_number}", True)
        add_field(embed, "Container", f"`{container_name}`", True)
        add_field(embed, "Reason", reason, False)
        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(embed=create_error_embed("Deletion Failed", f"Error: {str(e)}"))

@bot.command(name='add-resources')
@is_admin()
async def add_resources(ctx, vps_id: str, ram: int = None, cpu: int = None, disk: int = None):
    """Add resources to a BharatCloud VPS (Admin only)"""
    if ram is None and cpu is None and disk is None:
        await ctx.send(embed=create_error_embed("Missing Parameters", "Please specify at least one resource to add (ram, cpu, or disk)"))
        return
    
    # Find the VPS in our database
    found_vps = None
    user_id = None
    vps_index = None
    
    for uid, vps_list in vps_data.items():
        for i, vps in enumerate(vps_list):
            if vps['container_name'] == vps_id:
                found_vps = vps
                user_id = uid
                vps_index = i
                break
        if found_vps:
            break
    
    if not found_vps:
        await ctx.send(embed=create_error_embed("VPS Not Found", f"No BharatCloud VPS found with ID: `{vps_id}`"))
        return
    
    was_running = found_vps.get('status') == 'running' and not found_vps.get('suspended', False)
    if was_running:
        await ctx.send(embed=create_info_embed("Stopping VPS", f"Stopping BharatCloud VPS `{vps_id}` to apply resource changes..."))
        try:
            await execute_lxc(f"lxc stop {vps_id}")
            found_vps['status'] = 'stopped'
            save_data()
        except Exception as e:
            await ctx.send(embed=create_error_embed("Stop Failed", f"Error stopping VPS: {str(e)}"))
            return
    
    changes = []
    
    try:
        current_ram_gb = int(found_vps['ram'].replace('GB', ''))
        current_cpu = int(found_vps['cpu'])
        current_disk_gb = int(found_vps['storage'].replace('GB', ''))
        
        new_ram_gb = current_ram_gb
        new_cpu = current_cpu
        new_disk_gb = current_disk_gb
        
        # Add RAM if specified
        if ram is not None and ram > 0:
            new_ram_gb += ram
            ram_mb = new_ram_gb * 1024
            await execute_lxc(f"lxc config set {vps_id} limits.memory {ram_mb}MB")
            changes.append(f"RAM: +{ram}GB (New total: {new_ram_gb}GB)")
        
        # Add CPU if specified
        if cpu is not None and cpu > 0:
            new_cpu += cpu
            await execute_lxc(f"lxc config set {vps_id} limits.cpu {new_cpu}")
            changes.append(f"CPU: +{cpu} cores (New total: {new_cpu} cores)")
        
        # Add disk if specified
        if disk is not None and disk > 0:
            new_disk_gb += disk
            await execute_lxc(f"lxc config device set {vps_id} root size {new_disk_gb}GB")
            changes.append(f"Disk: +{disk}GB (New total: {new_disk_gb}GB)")
        
        # Update VPS data
        found_vps['ram'] = f"{new_ram_gb}GB"
        found_vps['cpu'] = str(new_cpu)
        found_vps['storage'] = f"{new_disk_gb}GB"
        found_vps['config'] = f"{new_ram_gb}GB RAM / {new_cpu} CPU / {new_disk_gb}GB Disk"
        
        # Save changes to database
        vps_data[user_id][vps_index] = found_vps
        save_data()
        
        # Start the VPS if it was running before
        if was_running:
            await execute_lxc(f"lxc start {vps_id}")
            found_vps['status'] = 'running'
            save_data()
        
        embed = create_success_embed("Resources Added", f"Successfully added resources to BharatCloud VPS `{vps_id}`")
        add_field(embed, "Changes Applied", "\n".join(changes), False)
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(embed=create_error_embed("Resource Addition Failed", f"Error: {str(e)}"))

@bot.command(name='admin-add')
@is_main_admin()
async def admin_add(ctx, user: discord.Member):
    """Add BharatCloud admin (Main admin only)"""
    user_id = str(user.id)
    if user_id == str(MAIN_ADMIN_ID):
        await ctx.send(embed=create_error_embed("Already Admin", "This user is already the main BharatCloud admin!"))
        return

    if user_id in admin_data.get("admins", []):
        await ctx.send(embed=create_error_embed("Already Admin", f"{user.mention} is already a BharatCloud admin!"))
        return

    if "admins" not in admin_data:
        admin_data["admins"] = []

    admin_data["admins"].append(user_id)
    save_data()
    await ctx.send(embed=create_success_embed("Admin Added", f"{user.mention} is now a BharatCloud admin!"))
    try:
        await user.send(embed=create_embed("🎉 BharatCloud Admin Role Granted", f"You are now a BharatCloud admin by {ctx.author.mention}", 0x00ff88))
    except discord.Forbidden:
        await ctx.send(embed=create_info_embed("Notification Failed", f"Could not DM {user.mention}"))

@bot.command(name='admin-remove')
@is_main_admin()
async def admin_remove(ctx, user: discord.Member):
    """Remove BharatCloud admin (Main admin only)"""
    user_id = str(user.id)
    if user_id == str(MAIN_ADMIN_ID):
        await ctx.send(embed=create_error_embed("Cannot Remove", "You cannot remove the main BharatCloud admin!"))
        return

    if user_id not in admin_data.get("admins", []):
        await ctx.send(embed=create_error_embed("Not Admin", f"{user.mention} is not a BharatCloud admin!"))
        return

    admin_data["admins"].remove(user_id)
    save_data()
    await ctx.send(embed=create_success_embed("Admin Removed", f"{user.mention} is no longer a BharatCloud admin!"))
    try:
        await user.send(embed=create_embed("⚠️ BharatCloud Admin Role Revoked", f"Your admin role was removed by {ctx.author.mention}", 0xff3366))
    except discord.Forbidden:
        await ctx.send(embed=create_info_embed("Notification Failed", f"Could not DM {user.mention}"))

@bot.command(name='admin-list')
@is_main_admin()
async def admin_list(ctx):
    """List all BharatCloud admins (Main admin only)"""
    admins = admin_data.get("admins", [])
    main_admin = await bot.fetch_user(MAIN_ADMIN_ID)

    embed = create_embed("👑 BharatCloud Admin Team", "Current BharatCloud administrators:", 0x1a1a1a)
    add_field(embed, "🔰 Main Admin", f"{main_admin.mention} (ID: {MAIN_ADMIN_ID})", False)

    if admins:
        admin_list = []
        for admin_id in admins:
            try:
                admin_user = await bot.fetch_user(int(admin_id))
                admin_list.append(f"• {admin_user.mention} (ID: {admin_id})")
            except:
                admin_list.append(f"• Unknown User (ID: {admin_id})")

        admin_text = "\n".join(admin_list)
        add_field(embed, "🛡️ Admins", admin_text, False)
    else:
        add_field(embed, "🛡️ Admins", "No additional BharatCloud admins", False)

    await ctx.send(embed=embed)

@bot.command(name='userinfo')
@is_admin()
async def user_info(ctx, user: discord.Member):
    """Get detailed information about a BharatCloud user (Admin only)"""
    user_id = str(user.id)

    # Get user's VPS
    vps_list = vps_data.get(user_id, [])

    embed = create_embed(f"BharatCloud User Information - {user.name}", f"Detailed information for {user.mention}", 0x1a1a1a)

    # User details
    add_field(embed, "👤 User Details", f"**Name:** {user.name}\n**ID:** {user.id}\n**Joined:** {user.joined_at.strftime('%Y-%m-%d %H:%M:%S')}", False)

    # VPS info
    if vps_list:
        vps_info = []
        total_ram = 0
        total_cpu = 0
        total_storage = 0
        running_count = 0
        suspended_count = 0

        for i, vps in enumerate(vps_list):
            status_emoji = "🟢" if vps.get('status') == 'running' and not vps.get('suspended', False) else "🟡" if vps.get('suspended', False) else "🔴"
            status_text = vps.get('status', 'unknown').upper()
            if vps.get('suspended', False):
                status_text += " (SUSPENDED)"
                suspended_count += 1
            else:
                running_count += 1 if vps.get('status') == 'running' else 0
            vps_info.append(f"{status_emoji} VPS {i+1}: `{vps['container_name']}` - {status_text}")

            # Calculate totals
            ram_gb = int(vps['ram'].replace('GB', ''))
            storage_gb = int(vps['storage'].replace('GB', ''))
            total_ram += ram_gb
            total_cpu += int(vps['cpu'])
            total_storage += storage_gb

        vps_summary = f"**Total VPS:** {len(vps_list)}\n**Running:** {running_count}\n**Suspended:** {suspended_count}\n**Total RAM:** {total_ram}GB\n**Total CPU:** {total_cpu} cores\n**Total Storage:** {total_storage}GB"
        add_field(embed, "🖥️ BharatCloud VPS Information", vps_summary, False)
        
        # Create additional embeds if VPS list is too long
        if len(vps_info) > 10:
            # First embed with first 10 VPS
            first_embed = embed
            add_field(first_embed, "📋 VPS List (1-10)", "\n".join(vps_info[:10]), False)
            await ctx.send(embed=first_embed)
            
            # Additional embeds for remaining VPS
            for i in range(10, len(vps_info), 10):
                chunk = vps_info[i:i+10]
                additional_embed = create_embed(f"BharatCloud VPS List ({i+1}-{min(i+10, len(vps_info))})", f"More VPS for {user.mention}", 0x1a1a1a)
                add_field(additional_embed, "📋 VPS List", "\n".join(chunk), False)
                await ctx.send(embed=additional_embed)
        else:
            add_field(embed, "📋 VPS List", "\n".join(vps_info), False)
            await ctx.send(embed=embed)
    else:
        add_field(embed, "🖥️ BharatCloud VPS Information", "**No VPS owned**", False)
        await ctx.send(embed=embed)

    # Check if user is admin
    is_admin_user = user_id == str(MAIN_ADMIN_ID) or user_id in admin_data.get("admins", [])
    add_field(embed, "🛡️ BharatCloud Admin Status", f"**{'Yes' if is_admin_user else 'No'}**", False)

@bot.command(name='serverstats')
@is_admin()
async def server_stats(ctx):
    """Show BharatCloud server statistics (Admin only)"""
    total_users = len(vps_data)
    total_vps = sum(len(vps_list) for vps_list in vps_data.values())

    # Calculate resources
    total_ram = 0
    total_cpu = 0
    total_storage = 0
    running_vps = 0
    suspended_vps = 0

    for vps_list in vps_data.values():
        for vps in vps_list:
            ram_gb = int(vps['ram'].replace('GB', ''))
            storage_gb = int(vps['storage'].replace('GB', ''))
            total_ram += ram_gb
            total_cpu += int(vps['cpu'])
            total_storage += storage_gb
            if vps.get('status') == 'running':
                if vps.get('suspended', False):
                    suspended_vps += 1
                else:
                    running_vps += 1

    embed = create_embed("📊 BharatCloud Server Statistics", "Current BharatCloud server overview", 0x1a1a1a)
    add_field(embed, "👥 Users", f"**Total Users:** {total_users}\n**Total Admins:** {len(admin_data.get('admins', [])) + 1}", False)
    add_field(embed, "🖥️ VPS", f"**Total VPS:** {total_vps}\n**Running:** {running_vps}\n**Suspended:** {suspended_vps}\n**Stopped:** {total_vps - running_vps - suspended_vps}", False)
    add_field(embed, "📈 Resources", f"**Total RAM:** {total_ram}GB\n**Total CPU:** {total_cpu} cores\n**Total Storage:** {total_storage}GB", False)

    await ctx.send(embed=embed)

@bot.command(name='vpsinfo')
@is_admin()
async def vps_info(ctx, container_name: str = None):
    """Get detailed BharatCloud VPS information (Admin only)"""
    if not container_name:
        # Show all VPS
        all_vps = []
        for user_id, vps_list in vps_data.items():
            try:
                user = await bot.fetch_user(int(user_id))
                for i, vps in enumerate(vps_list):
                    status_text = vps.get('status', 'unknown').upper()
                    if vps.get('suspended', False):
                        status_text += " (SUSPENDED)"
                    all_vps.append(f"**{user.name}** - BharatCloud VPS {i+1}: `{vps['container_name']}` - {status_text}")
            except:
                pass

        # Create multiple embeds if needed to avoid character limit
        for i in range(0, len(all_vps), 20):
            chunk = all_vps[i:i+20]
            embed = create_embed(f"🖥️ All BharatCloud VPS ({i+1}-{min(i+20, len(all_vps))})", f"List of all BharatCloud VPS deployments", 0x1a1a1a)
            add_field(embed, "VPS List", "\n".join(chunk), False)
            await ctx.send(embed=embed)
    else:
        # Show specific VPS info
        found_vps = None
        found_user = None

        for user_id, vps_list in vps_data.items():
            for vps in vps_list:
                if vps['container_name'] == container_name:
                    found_vps = vps
                    found_user = await bot.fetch_user(int(user_id))
                    break
            if found_vps:
                break

        if not found_vps:
            await ctx.send(embed=create_error_embed("VPS Not Found", f"No BharatCloud VPS found with container name: `{container_name}`"))
            return

        suspended_text = " (SUSPENDED)" if found_vps.get('suspended', False) else ""
        embed = create_embed(f"🖥️ BharatCloud VPS Information - {container_name}", f"Details for VPS owned by {found_user.mention}{suspended_text}", 0x1a1a1a)
        add_field(embed, "👤 Owner", f"**Name:** {found_user.name}\n**ID:** {found_user.id}", False)
        add_field(embed, "📊 Specifications", f"**RAM:** {found_vps['ram']}\n**CPU:** {found_vps['cpu']} Cores\n**Storage:** {found_vps['storage']}", False)
        add_field(embed, "📈 Status", f"**Current:** {found_vps.get('status', 'unknown').upper()}{suspended_text}\n**Suspended:** {found_vps.get('suspended', False)}\n**Created:** {found_vps.get('created_at', 'Unknown')}", False)

        if 'config' in found_vps:
            add_field(embed, "⚙️ Configuration", f"**Config:** {found_vps['config']}", False)

        if found_vps.get('shared_with'):
            shared_users = []
            for shared_id in found_vps['shared_with']:
                try:
                    shared_user = await bot.fetch_user(int(shared_id))
                    shared_users.append(f"• {shared_user.mention}")
                except:
                    shared_users.append(f"• Unknown User ({shared_id})")
            shared_text = "\n".join(shared_users)
            add_field(embed, "🔗 Shared With", shared_text, False)

        await ctx.send(embed=embed)

@bot.command(name='restart-vps')
@is_admin()
async def restart_vps(ctx, container_name: str):
    """Restart a BharatCloud VPS (Admin only)"""
    await ctx.send(embed=create_info_embed("Restarting VPS", f"Restarting BharatCloud VPS `{container_name}`..."))

    try:
        await execute_lxc(f"lxc restart {container_name}")

        # Update status in database
        for user_id, vps_list in vps_data.items():
            for vps in vps_list:
                if vps['container_name'] == container_name:
                    vps['status'] = 'running'
                    vps['suspended'] = False
                    save_data()
                    break

        await ctx.send(embed=create_success_embed("VPS Restarted", f"BharatCloud VPS `{container_name}` has been restarted successfully!"))

    except Exception as e:
        await ctx.send(embed=create_error_embed("Restart Failed", f"Error: {str(e)}"))

@bot.command(name='backup-vps')
@is_admin()
async def backup_vps(ctx, container_name: str):
    """Create a snapshot of a BharatCloud VPS (Admin only)"""
    snapshot_name = f"BharatCloud-{container_name}-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    await ctx.send(embed=create_info_embed("Creating BharatCloud Backup", f"Creating snapshot of `{container_name}`..."))

    try:
        await execute_lxc(f"lxc snapshot {container_name} {snapshot_name}")
        await ctx.send(embed=create_success_embed("Backup Created", f"BharatCloud Snapshot `{snapshot_name}` created successfully!"))

    except Exception as e:
        await ctx.send(embed=create_error_embed("Backup Failed", f"Error: {str(e)}"))

@bot.command(name='restore-vps')
@is_admin()
async def restore_vps(ctx, container_name: str, snapshot_name: str):
    """Restore a BharatCloud VPS from snapshot (Admin only)"""
    await ctx.send(embed=create_info_embed("Restoring VPS", f"Restoring `{container_name}` from BharatCloud snapshot `{snapshot_name}`..."))

    try:
        await execute_lxc(f"lxc restore {container_name} {snapshot_name}")
        await ctx.send(embed=create_success_embed("VPS Restored", f"BharatCloud VPS `{container_name}` has been restored from snapshot!"))

    except Exception as e:
        await ctx.send(embed=create_error_embed("Restore Failed", f"Error: {str(e)}"))

@bot.command(name='list-snapshots')
@is_admin()
async def list_snapshots(ctx, container_name: str):
    """List all snapshots for a BharatCloud VPS (Admin only)"""
    try:
        # Improved parsing for lxc list --type snapshot
        proc = await asyncio.create_subprocess_exec(
            "lxc", "list", "--type", "snapshot", "--columns", "n",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise Exception(stderr.decode())

        snapshots = [line.strip() for line in stdout.decode().splitlines() if line.strip() and container_name in line]
        snapshots = [snap.split()[0] for snap in snapshots if snap]  # Extract names

        if snapshots:
            # Create multiple embeds if needed to avoid character limit
            for i in range(0, len(snapshots), 20):
                chunk = snapshots[i:i+20]
                embed = create_embed(f"📸 BharatCloud Snapshots for {container_name} ({i+1}-{min(i+20, len(snapshots))})", f"List of snapshots", 0x1a1a1a)
                add_field(embed, "Snapshots", "\n".join([f"• {snap}" for snap in chunk]), False)
                await ctx.send(embed=embed)
        else:
            await ctx.send(embed=create_info_embed("No Snapshots", f"No BharatCloud snapshots found for `{container_name}`"))

    except Exception as e:
        await ctx.send(embed=create_error_embed("Error", f"Error listing snapshots: {str(e)}"))

@bot.command(name='exec')
@is_admin()
async def execute_command(ctx, container_name: str, *, command: str):
    """Execute a command inside a BharatCloud VPS (Admin only)"""
    await ctx.send(embed=create_info_embed("Executing Command", f"Running command in BharatCloud VPS `{container_name}`..."))

    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "exec", container_name, "--", "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        output = stdout.decode() if stdout else "No output"
        error = stderr.decode() if stderr else ""

        embed = create_embed(f"Command Output - {container_name}", f"Command: `{command}`", 0x1a1a1a)

        if output.strip():
            # Split output if too long
            if len(output) > 1000:
                output = output[:1000] + "\n... (truncated)"
            add_field(embed, "📤 Output", f"```\n{output}\n```", False)

        if error.strip():
            if len(error) > 1000:
                error = error[:1000] + "\n... (truncated)"
            add_field(embed, "⚠️ Error", f"```\n{error}\n```", False)

        add_field(embed, "🔄 Exit Code", f"**{proc.returncode}**", False)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(embed=create_error_embed("Execution Failed", f"Error: {str(e)}"))

@bot.command(name='stop-vps-all')
@is_admin()
async def stop_all_vps(ctx):
    """Stop all BharatCloud VPS using lxc stop --all --force (Admin only)"""
    await ctx.send(embed=create_warning_embed("Stopping All BharatCloud VPS", "⚠️ **WARNING:** This will stop ALL running VPS on the BharatCloud server.\n\nThis action cannot be undone. Continue?"))

    class ConfirmView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=60)

        @discord.ui.button(label="Stop All VPS", style=discord.ButtonStyle.danger)
        async def confirm(self, interaction: discord.Interaction, item: discord.ui.Button):
            await interaction.response.defer()

            try:
                # Execute the lxc stop --all --force command
                proc = await asyncio.create_subprocess_exec(
                    "lxc", "stop", "--all", "--force",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    # Update all VPS status in database to stopped
                    stopped_count = 0
                    for user_id, vps_list in vps_data.items():
                        for vps in vps_list:
                            if vps.get('status') == 'running':
                                vps['status'] = 'stopped'
                                vps['suspended'] = False
                                stopped_count += 1

                    save_data()

                    embed = create_success_embed("All BharatCloud VPS Stopped", f"Successfully stopped {stopped_count} VPS using `lxc stop --all --force`")
                    output_text = stdout.decode() if stdout else 'No output'
                    add_field(embed, "Command Output", f"```\n{output_text}\n```", False)
                    await interaction.followup.send(embed=embed)
                else:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    embed = create_error_embed("Stop Failed", f"Failed to stop BharatCloud VPS: {error_msg}")
                    await interaction.followup.send(embed=embed)

            except Exception as e:
                embed = create_error_embed("Error", f"Error stopping VPS: {str(e)}")
                await interaction.followup.send(embed=embed)

        @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, item: discord.ui.Button):
            await interaction.response.edit_message(embed=create_info_embed("Operation Cancelled", "The stop all BharatCloud VPS operation has been cancelled."))

    await ctx.send(view=ConfirmView())

@bot.command(name='cpu-monitor')
@is_admin()
async def cpu_monitor_control(ctx, action: str = "status"):
    """Control BharatCloud CPU monitoring system (Admin only)"""
    global cpu_monitor_active
    
    if action.lower() == "status":
        status = "Active" if cpu_monitor_active else "Inactive"
        embed = create_embed("BharatCloud CPU Monitor Status", f"BharatCloud CPU monitoring is currently **{status}**", 0x00ccff if cpu_monitor_active else 0xffaa00)
        add_field(embed, "Threshold", f"{CPU_THRESHOLD}% CPU usage", True)
        add_field(embed, "Check Interval", f"60 seconds (host)", True)
        await ctx.send(embed=embed)
    elif action.lower() == "enable":
        cpu_monitor_active = True
        await ctx.send(embed=create_success_embed("CPU Monitor Enabled", "BharatCloud CPU monitoring has been enabled."))
    elif action.lower() == "disable":
        cpu_monitor_active = False
        await ctx.send(embed=create_warning_embed("CPU Monitor Disabled", "BharatCloud CPU monitoring has been disabled."))
    else:
        await ctx.send(embed=create_error_embed("Invalid Action", "Use: `!cpu-monitor <status|enable|disable>`"))

@bot.command(name='resize-vps')
@is_admin()
async def resize_vps(ctx, container_name: str, ram: int = None, cpu: int = None, disk: int = None):
    """Resize BharatCloud VPS resources (Admin only)"""
    if ram is None and cpu is None and disk is None:
        await ctx.send(embed=create_error_embed("Missing Parameters", "Please specify at least one resource to resize (ram, cpu, or disk)"))
        return
    
    # Find the VPS in our database
    found_vps = None
    user_id = None
    vps_index = None
    
    for uid, vps_list in vps_data.items():
        for i, vps in enumerate(vps_list):
            if vps['container_name'] == container_name:
                found_vps = vps
                user_id = uid
                vps_index = i
                break
        if found_vps:
            break
    
    if not found_vps:
        await ctx.send(embed=create_error_embed("VPS Not Found", f"No BharatCloud VPS found with container name: `{container_name}`"))
        return
    
    was_running = found_vps.get('status') == 'running' and not found_vps.get('suspended', False)
    if was_running:
        await ctx.send(embed=create_info_embed("Stopping VPS", f"Stopping BharatCloud VPS `{container_name}` to apply resource changes..."))
        try:
            await execute_lxc(f"lxc stop {container_name}")
            found_vps['status'] = 'stopped'
            save_data()
        except Exception as e:
            await ctx.send(embed=create_error_embed("Stop Failed", f"Error stopping VPS: {str(e)}"))
            return
    
    changes = []
    
    try:
        new_ram = int(found_vps['ram'].replace('GB', ''))
        new_cpu = int(found_vps['cpu'])
        new_disk = int(found_vps['storage'].replace('GB', ''))
        
        # Resize RAM if specified
        if ram is not None and ram > 0:
            new_ram = ram
            ram_mb = ram * 1024
            await execute_lxc(f"lxc config set {container_name} limits.memory {ram_mb}MB")
            changes.append(f"RAM: {ram}GB")
        
        # Resize CPU if specified
        if cpu is not None and cpu > 0:
            new_cpu = cpu
            await execute_lxc(f"lxc config set {container_name} limits.cpu {cpu}")
            changes.append(f"CPU: {cpu} cores")
        
        # Resize disk if specified
        if disk is not None and disk > 0:
            new_disk = disk
            await execute_lxc(f"lxc config device set {container_name} root size {disk}GB")
            changes.append(f"Disk: {disk}GB")
        
        # Update VPS data
        found_vps['ram'] = f"{new_ram}GB"
        found_vps['cpu'] = str(new_cpu)
        found_vps['storage'] = f"{new_disk}GB"
        found_vps['config'] = f"{new_ram}GB RAM / {new_cpu} CPU / {new_disk}GB Disk"
        
        # Save changes to database
        vps_data[user_id][vps_index] = found_vps
        save_data()
        
        # Start the VPS if it was running before
        if was_running:
            await execute_lxc(f"lxc start {container_name}")
            found_vps['status'] = 'running'
            save_data()
        
        embed = create_success_embed("VPS Resized", f"Successfully resized resources for BharatCloud VPS `{container_name}`")
        add_field(embed, "Changes Applied", "\n".join(changes), False)
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(embed=create_error_embed("Resize Failed", f"Error: {str(e)}"))

@bot.command(name='clone-vps')
@is_admin()
async def clone_vps(ctx, container_name: str, new_name: str = None):
    """Clone a BharatCloud VPS (Admin only)"""
    if not new_name:
        # Generate a new name if not provided
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        new_name = f"BharatCloud-{container_name}-clone-{timestamp}"
    
    await ctx.send(embed=create_info_embed("Cloning VPS", f"Cloning BharatCloud VPS `{container_name}` to `{new_name}`..."))
    
    try:
        # Find the original VPS in our database
        found_vps = None
        user_id = None
        
        for uid, vps_list in vps_data.items():
            for vps in vps_list:
                if vps['container_name'] == container_name:
                    found_vps = vps
                    user_id = uid
                    break
            if found_vps:
                break
        
        if not found_vps:
            await ctx.send(embed=create_error_embed("VPS Not Found", f"No BharatCloud VPS found with container name: `{container_name}`"))
            return
        
        # Clone the container
        await execute_lxc(f"lxc copy {container_name} {new_name}")
        
        # Start the new container
        await execute_lxc(f"lxc start {new_name}")
        
        # Create a new VPS entry in the database
        if user_id not in vps_data:
            vps_data[user_id] = []
        
        new_vps = found_vps.copy()
        new_vps['container_name'] = new_name
        new_vps['status'] = 'running'
        new_vps['suspended'] = False
        new_vps['suspension_history'] = []
        new_vps['created_at'] = datetime.now().isoformat()
        new_vps['shared_with'] = []
        
        vps_data[user_id].append(new_vps)
        save_data()
        
        embed = create_success_embed("VPS Cloned", f"Successfully cloned BharatCloud VPS `{container_name}` to `{new_name}`")
        add_field(embed, "New VPS Details", f"**RAM:** {new_vps['ram']}\n**CPU:** {new_vps['cpu']} Cores\n**Storage:** {new_vps['storage']}", False)
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(embed=create_error_embed("Clone Failed", f"Error: {str(e)}"))

@bot.command(name='migrate-vps')
@is_admin()
async def migrate_vps(ctx, container_name: str, target_pool: str):
    """Migrate a BharatCloud VPS to a different storage pool (Admin only)"""
    await ctx.send(embed=create_info_embed("Migrating VPS", f"Migrating BharatCloud VPS `{container_name}` to storage pool `{target_pool}`..."))
    
    try:
        # Stop the container first
        await execute_lxc(f"lxc stop {container_name}")
        
        # Create a temporary name for migration
        temp_name = f"BharatCloud-{container_name}-temp-{int(time.time())}"
        
        # Copy to new pool with temp name
        await execute_lxc(f"lxc copy {container_name} {temp_name} --storage {target_pool}")
        
        # Delete the old container
        await execute_lxc(f"lxc delete {container_name} --force")
        
        # Rename temp to original name
        await execute_lxc(f"lxc rename {temp_name} {container_name}")
        
        # Start the container again
        await execute_lxc(f"lxc start {container_name}")
        
        # Update status in database
        for user_id, vps_list in vps_data.items():
            for vps in vps_list:
                if vps['container_name'] == container_name:
                    vps['status'] = 'running'
                    vps['suspended'] = False
                    save_data()
                    break
        
        await ctx.send(embed=create_success_embed("VPS Migrated", f"Successfully migrated BharatCloud VPS `{container_name}` to storage pool `{target_pool}`"))
        
    except Exception as e:
        await ctx.send(embed=create_error_embed("Migration Failed", f"Error: {str(e)}"))

@bot.command(name='vps-stats')
@is_admin()
async def vps_stats(ctx, container_name: str):
    """Show detailed resource usage statistics for a BharatCloud VPS (Admin only)"""
    await ctx.send(embed=create_info_embed("Gathering Statistics", f"Collecting statistics for BharatCloud VPS `{container_name}`..."))
    
    try:
        status = await get_container_status(container_name)
        cpu_usage = await get_container_cpu(container_name)
        memory_usage = await get_container_memory(container_name)
        disk_usage = await get_container_disk(container_name)
        network_usage = "N/A"  # Simplified for now
        
        # Create embed with statistics
        embed = create_embed(f"📊 BharatCloud VPS Statistics - {container_name}", f"Resource usage statistics", 0x1a1a1a)
        add_field(embed, "📈 Status", f"**{status}**", False)
        add_field(embed, "💻 CPU Usage", f"**{cpu_usage}**", True)
        add_field(embed, "🧠 Memory Usage", f"**{memory_usage}**", True)
        add_field(embed, "💾 Disk Usage", f"**{disk_usage}**", True)
        add_field(embed, "🌐 Network Usage", f"**{network_usage}**", False)
        
        # Find the VPS in our database
        found_vps = None
        for vps_list in vps_data.values():
            for vps in vps_list:
                if vps['container_name'] == container_name:
                    found_vps = vps
                    break
            if found_vps:
                break
        
        if found_vps:
            suspended_text = " (SUSPENDED)" if found_vps.get('suspended', False) else ""
            add_field(embed, "📋 Allocated Resources", 
                           f"**RAM:** {found_vps['ram']}\n**CPU:** {found_vps['cpu']} Cores\n**Storage:** {found_vps['storage']}\n**Status:** {found_vps.get('status', 'unknown').upper()}{suspended_text}", 
                           False)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(embed=create_error_embed("Statistics Failed", f"Error: {str(e)}"))

@bot.command(name='vps-network')
@is_admin()
async def vps_network(ctx, container_name: str, action: str, value: str = None):
    """Manage BharatCloud VPS network settings (Admin only)"""
    if action.lower() not in ["list", "add", "remove", "limit"]:
        await ctx.send(embed=create_error_embed("Invalid Action", "Use: `!vps-network <container> <list|add|remove|limit> [value]`"))
        return
    
    try:
        if action.lower() == "list":
            # List network interfaces
            proc = await asyncio.create_subprocess_exec(
                "lxc", "exec", container_name, "--", "ip", "addr",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                output = stdout.decode()
                # Split output if too long
                if len(output) > 1000:
                    output = output[:1000] + "\n... (truncated)"
                
                embed = create_embed(f"🌐 BharatCloud Network Interfaces - {container_name}", "Network configuration", 0x1a1a1a)
                add_field(embed, "Interfaces", f"```\n{output}\n```", False)
                await ctx.send(embed=embed)
            else:
                await ctx.send(embed=create_error_embed("Error", f"Failed to list network interfaces: {stderr.decode()}"))
        
        elif action.lower() == "limit" and value:
            # Set network limit
            await execute_lxc(f"lxc config device set {container_name} eth0 limits.egress {value}")
            await execute_lxc(f"lxc config device set {container_name} eth0 limits.ingress {value}")
            await ctx.send(embed=create_success_embed("Network Limited", f"Set BharatCloud network limit to {value} for `{container_name}`"))
        
        elif action.lower() in ["add", "remove"]:
            await ctx.send(embed=create_info_embed("Not Implemented", f"BharatCloud Network {action} is not yet implemented. Use list or limit for now."))
        
        else:
            await ctx.send(embed=create_error_embed("Invalid Parameters", "Please provide valid parameters for the action"))
    
    except Exception as e:
        await ctx.send(embed=create_error_embed("Network Management Failed", f"Error: {str(e)}"))

@bot.command(name='vps-processes')
@is_admin()
async def vps_processes(ctx, container_name: str):
    """Show running processes in a BharatCloud VPS (Admin only)"""
    await ctx.send(embed=create_info_embed("Gathering Processes", f"Listing processes in BharatCloud VPS `{container_name}`..."))
    
    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "exec", container_name, "--", "ps", "aux",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            output = stdout.decode()
            # Split output if too long
            if len(output) > 1000:
                output = output[:1000] + "\n... (truncated)"
            
            embed = create_embed(f"⚙️ BharatCloud Processes - {container_name}", "Running processes", 0x1a1a1a)
            add_field(embed, "Process List", f"```\n{output}\n```", False)
            await ctx.send(embed=embed)
        else:
            await ctx.send(embed=create_error_embed("Error", f"Failed to list processes: {stderr.decode()}"))
    
    except Exception as e:
        await ctx.send(embed=create_error_embed("Process Listing Failed", f"Error: {str(e)}"))

@bot.command(name='vps-logs')
@is_admin()
async def vps_logs(ctx, container_name: str, lines: int = 50):
    """Show recent logs from a BharatCloud VPS (Admin only)"""
    await ctx.send(embed=create_info_embed("Gathering Logs", f"Fetching last {lines} lines from BharatCloud VPS `{container_name}`..."))
    
    try:
        proc = await asyncio.create_subprocess_exec(
            "lxc", "exec", container_name, "--", "journalctl", "-n", str(lines),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            output = stdout.decode()
            # Split output if too long
            if len(output) > 1000:
                output = output[:1000] + "\n... (truncated)"
            
            embed = create_embed(f"📋 BharatCloud Logs - {container_name}", f"Last {lines} log lines", 0x1a1a1a)
            add_field(embed, "System Logs", f"```\n{output}\n```", False)
            await ctx.send(embed=embed)
        else:
            await ctx.send(embed=create_error_embed("Error", f"Failed to fetch logs: {stderr.decode()}"))
    
    except Exception as e:
        await ctx.send(embed=create_error_embed("Log Retrieval Failed", f"Error: {str(e)}"))

@bot.command(name='suspend-vps')
@is_admin()
async def suspend_vps(ctx, container_name: str, *, reason: str = "Admin action"):
    """Suspend a BharatCloud VPS (Admin only)"""
    found = False
    for uid, lst in vps_data.items():
        for vps in lst:
            if vps['container_name'] == container_name:
                if vps.get('status') != 'running':
                    await ctx.send(embed=create_error_embed("Cannot Suspend", "BharatCloud VPS must be running to suspend."))
                    return
                try:
                    await execute_lxc(f"lxc stop {container_name}")
                    vps['status'] = 'suspended'
                    vps['suspended'] = True
                    if 'suspension_history' not in vps:
                        vps['suspension_history'] = []
                    vps['suspension_history'].append({
                        'time': datetime.now().isoformat(),
                        'reason': reason,
                        'by': f"{ctx.author.name} ({ctx.author.id})"
                    })
                    save_data()
                except Exception as e:
                    await ctx.send(embed=create_error_embed("Suspend Failed", str(e)))
                    return
                # DM owner
                try:
                    owner = await bot.fetch_user(int(uid))
                    embed = create_warning_embed("🚨 BharatCloud VPS Suspended", f"Your VPS `{container_name}` has been suspended by an admin.\n\n**Reason:** {reason}\n\nContact a BharatCloud admin to unsuspend.")
                    await owner.send(embed=embed)
                except Exception as dm_e:
                    logger.error(f"Failed to DM owner {uid}: {dm_e}")
                await ctx.send(embed=create_success_embed("VPS Suspended", f"BharatCloud VPS `{container_name}` suspended. Reason: {reason}"))
                found = True
                break
        if found:
            break
    if not found:
        await ctx.send(embed=create_error_embed("Not Found", f"BharatCloud VPS `{container_name}` not found."))

@bot.command(name='unsuspend-vps')
@is_admin()
async def unsuspend_vps(ctx, container_name: str):
    """Unsuspend a BharatCloud VPS (Admin only)"""
    found = False
    for uid, lst in vps_data.items():
        for vps in lst:
            if vps['container_name'] == container_name:
                if not vps.get('suspended', False):
                    await ctx.send(embed=create_error_embed("Not Suspended", "BharatCloud VPS is not suspended."))
                    return
                try:
                    vps['suspended'] = False
                    vps['status'] = 'running'
                    await execute_lxc(f"lxc start {container_name}")
                    save_data()
                    await ctx.send(embed=create_success_embed("VPS Unsuspended", f"BharatCloud VPS `{container_name}` unsuspended and started."))
                    found = True
                except Exception as e:
                    await ctx.send(embed=create_error_embed("Start Failed", str(e)))
                break
        if found:
            break
    if not found:
        await ctx.send(embed=create_error_embed("Not Found", f"BharatCloud VPS `{container_name}` not found."))

@bot.command(name='suspension-logs')
@is_admin()
async def suspension_logs(ctx, container_name: str = None):
    """View BharatCloud suspension logs (Admin only)"""
    if container_name:
        # Specific VPS
        found = None
        for lst in vps_data.values():
            for vps in lst:
                if vps['container_name'] == container_name:
                    found = vps
                    break
            if found:
                break
        if not found:
            await ctx.send(embed=create_error_embed("Not Found", f"BharatCloud VPS `{container_name}` not found."))
            return
        history = found.get('suspension_history', [])
        if not history:
            await ctx.send(embed=create_info_embed("No Suspensions", f"No BharatCloud suspension history for `{container_name}`."))
            return
        embed = create_embed("BharatCloud Suspension History", f"For `{container_name}`")
        text = []
        for h in sorted(history, key=lambda x: x['time'], reverse=True)[:10]:  # Last 10
            t = datetime.fromisoformat(h['time']).strftime('%Y-%m-%d %H:%M:%S')
            text.append(f"**{t}** - {h['reason']} (by {h['by']})")
        add_field(embed, "History", "\n".join(text), False)
        if len(history) > 10:
            add_field(embed, "Note", "Showing last 10 entries.")
        await ctx.send(embed=embed)
    else:
        # All logs
        all_logs = []
        for uid, lst in vps_data.items():
            for vps in lst:
                h = vps.get('suspension_history', [])
                for event in sorted(h, key=lambda x: x['time'], reverse=True):
                    t = datetime.fromisoformat(event['time']).strftime('%Y-%m-%d %H:%M')
                    all_logs.append(f"**{t}** - VPS `{vps['container_name']}` (Owner: <@{uid}>) - {event['reason']} (by {event['by']})")
        if not all_logs:
            await ctx.send(embed=create_info_embed("No Suspensions", "No BharatCloud suspension events recorded."))
            return
        # Split into embeds
        for i in range(0, len(all_logs), 10):
            chunk = all_logs[i:i+10]
            embed = create_embed(f"BharatCloud Suspension Logs ({i+1}-{min(i+10, len(all_logs))})", f"Global suspension events (newest first)")
            add_field(embed, "Events", "\n".join(chunk), False)
            await ctx.send(embed=embed)

@bot.command(name='help')
async def show_help(ctx):
    """Show BharatCloud help information"""
    user_id = str(ctx.author.id)
    is_user_admin = user_id == str(MAIN_ADMIN_ID) or user_id in admin_data.get("admins", [])
    is_user_main_admin = user_id == str(MAIN_ADMIN_ID)

    # Create multiple embeds for help to avoid character limit
    # First embed with user commands
    embed = create_embed("📚 BharatCloud Command Help - User Commands", "BharatCloud VPS Manager Commands:", 0x1a1a1a)

    user_commands = [
        ("!ping", "Check BharatCloud bot latency"),
        ("!uptime", "Show host uptime"),
        ("!myvps", "List your BharatCloud VPS"),
        ("!manage [@user]", "Manage your VPS or another user's VPS (Admin only)"),
        ("!share-user @user <vps_number>", "Share BharatCloud VPS access"),
        ("!share-ruser @user <vps_number>", "Revoke BharatCloud VPS access"),
        ("!manage-shared @owner <vps_number>", "Manage shared BharatCloud VPS")
    ]

    user_commands_text = "\n".join([f"**{cmd}** - {desc}" for cmd, desc in user_commands])
    add_field(embed, "👤 User Commands", user_commands_text, False)
    await ctx.send(embed=embed)

    if is_user_admin:
        # Second embed with admin commands
        embed = create_embed("📚 BharatCloud Command Help - Admin Commands", "BharatCloud VPS Manager Commands:", 0x1a1a1a)

        admin_commands = [
            ("!lxc-list", "List all LXC containers"),
            ("!create <ram_gb> <cpu_cores> <disk_gb> @user", "Create custom BharatCloud VPS"),
            ("!delete-vps @user <vps_number> <reason>", "Delete user's BharatCloud VPS"),
            ("!add-resources <vps_id> [ram] [cpu] [disk]", "Add resources to a BharatCloud VPS"),
            ("!resize-vps <container> [ram] [cpu] [disk]", "Resize BharatCloud VPS resources"),
            ("!suspend-vps <container> [reason]", "Suspend a BharatCloud VPS"),
            ("!unsuspend-vps <container>", "Unsuspend a BharatCloud VPS"),
            ("!suspension-logs [container]", "View BharatCloud suspension logs"),
            ("!userinfo @user", "Get detailed BharatCloud user information"),
            ("!serverstats", "Show BharatCloud server statistics"),
            ("!vpsinfo [container]", "Get BharatCloud VPS information"),
            ("!list-all", "View all BharatCloud VPS and user information"),
            ("!restart-vps <container>", "Restart a BharatCloud VPS"),
            ("!backup-vps <container>", "Create BharatCloud VPS snapshot"),
            ("!restore-vps <container> <snapshot>", "Restore from BharatCloud snapshot"),
            ("!list-snapshots <container>", "List BharatCloud VPS snapshots"),
            ("!exec <container> <command>", "Execute command in BharatCloud VPS"),
            ("!stop-vps-all", "Stop all BharatCloud VPS with lxc stop --all --force"),
            ("!cpu-monitor <status|enable|disable>", "Control BharatCloud CPU monitoring system"),
            ("!clone-vps <container> [new_name]", "Clone a BharatCloud VPS"),
            ("!migrate-vps <container> <pool>", "Migrate BharatCloud VPS to storage pool"),
            ("!vps-stats <container>", "Show BharatCloud VPS resource stats"),
            ("!vps-network <container> <action> [value]", "Manage BharatCloud network"),
            ("!vps-processes <container>", "List BharatCloud processes"),
            ("!vps-logs <container> [lines]", "Show BharatCloud system logs")
        ]

        admin_commands_text = "\n".join([f"**{cmd}** - {desc}" for cmd, desc in admin_commands])
        add_field(embed, "🛡️ Admin Commands", admin_commands_text, False)
        await ctx.send(embed=embed)

    if is_user_main_admin:
        # Third embed with main admin commands
        embed = create_embed("📚 BharatCloud Command Help - Main Admin Commands", "BharatCloud VPS Manager Commands:", 0x1a1a1a)

        main_admin_commands = [
            ("!admin-add @user", "Promote to BharatCloud admin"),
            ("!admin-remove @user", "Remove BharatCloud admin"),
            ("!admin-list", "View all BharatCloud admins")
        ]

        main_admin_commands_text = "\n".join([f"**{cmd}** - {desc}" for cmd, desc in main_admin_commands])
        add_field(embed, "👑 Main Admin Commands", main_admin_commands_text, False)
        embed.set_footer(text="BharatCloud VPS Manager • Auto-suspend on high usage • Enhanced monitoring")
        await ctx.send(embed=embed)

# Command aliases for typos
@bot.command(name='mangage')
async def manage_typo(ctx):
    """Handle typo in manage command"""
    await ctx.send(embed=create_info_embed("Command Correction", "Did you mean `!manage`? Use the correct BharatCloud command."))

@bot.command(name='stats')
async def stats_alias(ctx):
    """Alias for serverstats command"""
    if str(ctx.author.id) == str(MAIN_ADMIN_ID) or str(ctx.author.id) in admin_data.get("admins", []):
        await server_stats(ctx)
    else:
        await ctx.send(embed=create_error_embed("Access Denied", "This BharatCloud command requires admin privileges."))

@bot.command(name='info')
async def info_alias(ctx):
    """Alias for userinfo command"""
    if str(ctx.author.id) == str(MAIN_ADMIN_ID) or str(ctx.author.id) in admin_data.get("admins", []):
        await ctx.send(embed=create_error_embed("Usage", "Please specify a user: `!info @user`"))
    else:
        await ctx.send(embed=create_error_embed("Access Denied", "This BharatCloud command requires admin privileges."))

# Run the bot with your token
if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        logger.error("No Discord token found in DISCORD_TOKEN environment variable.")