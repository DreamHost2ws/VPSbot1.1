# bot_dedicated.py
# Dedicated VPS deployer variant of your original Lightplays bot
# IMPORTANT: revoke any public Discord token you shared and put a new token in .env

import discord
from discord.ext import commands
from discord import ui, app_commands
import os
import random
import string
import json
import subprocess
from dotenv import load_dotenv
import asyncio
import datetime
import docker
import time
import logging
import traceback
import aiohttp
import socket
import re
import psutil
import platform
import shutil
from typing import Optional
import sqlite3
import pickle

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lightplays_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LightplaysBot')

# load .env
load_dotenv()

# environment / defaults
TOKEN = os.getenv('DISCORD_TOKEN')
ADMIN_IDS = {int(id_) for id_ in os.getenv('ADMIN_IDS', '').split(',') if id_.strip()} or {1210291131301101618}
ADMIN_ROLE_ID = int(os.getenv('ADMIN_ROLE_ID', '1376177459870961694'))
WATERMARK = os.getenv('WATERMARK', "Lightplays VPS Service")
WELCOME_MESSAGE = os.getenv('WELCOME_MESSAGE', "Welcome To Lightplays! Get Started With Us!")
MAX_VPS_PER_USER = int(os.getenv('MAX_VPS_PER_USER', '3'))
DEFAULT_OS_IMAGE = os.getenv('DEFAULT_OS_IMAGE', 'arindamvm/unvm')  # should be SSH-enabled image
DOCKER_NETWORK = os.getenv('DOCKER_NETWORK', 'bridge')
MAX_CONTAINERS = int(os.getenv('MAX_CONTAINERS', '100'))
DB_FILE = os.getenv('DB_FILE', 'lightplays.db')
BACKUP_FILE = os.getenv('BACKUP_FILE', 'lightplays_backup.pkl')

# host SSH port allocation range (host ports to use for mapping container:22)
# Use an ephemeral port range that does not conflict with host services
SSH_PORT_RANGE = os.getenv('EXPOSE_SSH_PORT_RANGE', '20000-40000')
SSH_PORT_START, SSH_PORT_END = map(int, SSH_PORT_RANGE.split('-'))

# known miner patterns (unchanged)
MINER_PATTERNS = [
    'xmrig', 'ethminer', 'cgminer', 'sgminer', 'bfgminer',
    'minerd', 'cpuminer', 'cryptonight', 'stratum', 'pool'
]

# simple Dockerfile template kept if you want to build custom image
DOCKERFILE_TEMPLATE = """FROM {base_image}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y openssh-server sudo wget curl nano htop tmux net-tools iputils-ping && \
    mkdir /var/run/sshd && \
    echo 'root:{root_password}' | chpasswd && \
    useradd -m -s /bin/bash {username} && echo '{username}:{user_password}' | chpasswd && usermod -aG sudo {username} && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
"""

# ---------- Database class (kept mostly as-is) ----------
class Database:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._initialize_settings()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vps_instances (
                token TEXT PRIMARY KEY,
                vps_id TEXT UNIQUE,
                container_id TEXT,
                host_ssh_port INTEGER,
                memory INTEGER,
                cpu INTEGER,
                disk INTEGER,
                username TEXT,
                password TEXT,
                root_password TEXT,
                created_by TEXT,
                created_at TEXT,
                watermark TEXT,
                os_image TEXT,
                restart_count INTEGER DEFAULT 0,
                last_restart TEXT,
                status TEXT DEFAULT 'running',
                use_custom_image BOOLEAN DEFAULT 1
            )
        ''')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS usage_stats (key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS system_settings (key TEXT PRIMARY KEY, value TEXT)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS banned_users (user_id TEXT PRIMARY KEY)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS admin_users (user_id TEXT PRIMARY KEY)')
        self.conn.commit()

    def _initialize_settings(self):
        defaults = {
            'max_containers': str(MAX_CONTAINERS),
            'max_vps_per_user': str(MAX_VPS_PER_USER)
        }
        for key, value in defaults.items():
            self.cursor.execute('INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)', (key, value))
        self.cursor.execute('SELECT user_id FROM admin_users')
        for row in self.cursor.fetchall():
            ADMIN_IDS.add(int(row[0]))
        self.conn.commit()

    def get_setting(self, key, default=None):
        self.cursor.execute('SELECT value FROM system_settings WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return int(result[0]) if result else default

    def set_setting(self, key, value):
        self.cursor.execute('INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)', (key, str(value)))
        self.conn.commit()

    def get_stat(self, key, default=0):
        self.cursor.execute('SELECT value FROM usage_stats WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return result[0] if result else default

    def increment_stat(self, key, amount=1):
        current = self.get_stat(key)
        self.cursor.execute('INSERT OR REPLACE INTO usage_stats (key, value) VALUES (?, ?)', (key, current + amount))
        self.conn.commit()

    def get_vps_by_id(self, vps_id):
        self.cursor.execute('SELECT * FROM vps_instances WHERE vps_id = ?', (vps_id,))
        row = self.cursor.fetchone()
        if not row:
            return None, None
        columns = [desc[0] for desc in self.cursor.description]
        vps = dict(zip(columns, row))
        return vps['token'], vps

    def get_vps_by_token(self, token):
        self.cursor.execute('SELECT * FROM vps_instances WHERE token = ?', (token,))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def get_user_vps_count(self, user_id):
        self.cursor.execute('SELECT COUNT(*) FROM vps_instances WHERE created_by = ?', (str(user_id),))
        return self.cursor.fetchone()[0]

    def get_user_vps(self, user_id):
        self.cursor.execute('SELECT * FROM vps_instances WHERE created_by = ?', (str(user_id),))
        rows = self.cursor.fetchall()
        if not rows:
            return []
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, r)) for r in rows]

    def get_all_vps(self):
        self.cursor.execute('SELECT * FROM vps_instances')
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return {row[0]: dict(zip(columns, row)) for row in rows}

    def add_vps(self, vps_data):
        columns = ', '.join(vps_data.keys())
        placeholders = ', '.join('?' for _ in vps_data)
        self.cursor.execute(f'INSERT INTO vps_instances ({columns}) VALUES ({placeholders})', tuple(vps_data.values()))
        self.conn.commit()
        self.increment_stat('total_vps_created')

    def remove_vps(self, token):
        self.cursor.execute('DELETE FROM vps_instances WHERE token = ?', (token,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def update_vps(self, token, updates):
        set_clause = ', '.join(f'{k} = ?' for k in updates)
        values = list(updates.values()) + [token]
        self.cursor.execute(f'UPDATE vps_instances SET {set_clause} WHERE token = ?', values)
        self.conn.commit()
        return self.cursor.rowcount > 0

    def is_user_banned(self, user_id):
        self.cursor.execute('SELECT 1 FROM banned_users WHERE user_id = ?', (str(user_id),))
        return self.cursor.fetchone() is not None

    def ban_user(self, user_id):
        self.cursor.execute('INSERT OR IGNORE INTO banned_users (user_id) VALUES (?)', (str(user_id),))
        self.conn.commit()

    def unban_user(self, user_id):
        self.cursor.execute('DELETE FROM banned_users WHERE user_id = ?', (str(user_id),))
        self.conn.commit()

    def get_banned_users(self):
        self.cursor.execute('SELECT user_id FROM banned_users')
        return [row[0] for row in self.cursor.fetchall()]

    def add_admin(self, user_id):
        self.cursor.execute('INSERT OR IGNORE INTO admin_users (user_id) VALUES (?)', (str(user_id),))
        self.conn.commit()
        ADMIN_IDS.add(int(user_id))

    def remove_admin(self, user_id):
        self.cursor.execute('DELETE FROM admin_users WHERE user_id = ?', (str(user_id),))
        self.conn.commit()
        if int(user_id) in ADMIN_IDS:
            ADMIN_IDS.remove(int(user_id))

    def get_admins(self):
        self.cursor.execute('SELECT user_id FROM admin_users')
        return [row[0] for row in self.cursor.fetchall()]

    def backup_data(self):
        data = {
            'vps_instances': self.get_all_vps(),
            'usage_stats': {},
            'system_settings': {},
            'banned_users': self.get_banned_users(),
            'admin_users': self.get_admins()
        }
        self.cursor.execute('SELECT * FROM usage_stats')
        for row in self.cursor.fetchall():
            data['usage_stats'][row[0]] = row[1]
        self.cursor.execute('SELECT * FROM system_settings')
        for row in self.cursor.fetchall():
            data['system_settings'][row[0]] = row[1]
        with open(BACKUP_FILE, 'wb') as f:
            pickle.dump(data, f)
        return True

    def restore_data(self):
        if not os.path.exists(BACKUP_FILE):
            return False
        try:
            with open(BACKUP_FILE, 'rb') as f:
                data = pickle.load(f)
            self.cursor.execute('DELETE FROM vps_instances')
            self.cursor.execute('DELETE FROM usage_stats')
            self.cursor.execute('DELETE FROM system_settings')
            self.cursor.execute('DELETE FROM banned_users')
            self.cursor.execute('DELETE FROM admin_users')
            for token, vps in data['vps_instances'].items():
                columns = ', '.join(vps.keys())
                placeholders = ', '.join('?' for _ in vps)
                self.cursor.execute(f'INSERT INTO vps_instances ({columns}) VALUES ({placeholders})', tuple(vps.values()))
            for key, value in data['usage_stats'].items():
                self.cursor.execute('INSERT INTO usage_stats (key, value) VALUES (?, ?)', (key, value))
            for key, value in data['system_settings'].items():
                self.cursor.execute('INSERT INTO system_settings (key, value) VALUES (?, ?)', (key, value))
            for user_id in data['banned_users']:
                self.cursor.execute('INSERT INTO banned_users (user_id) VALUES (?)', (user_id,))
            for user_id in data['admin_users']:
                self.cursor.execute('INSERT INTO admin_users (user_id) VALUES (?)', (user_id,))
                ADMIN_IDS.add(int(user_id))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error restoring data: {e}")
            return False

    def close(self):
        self.conn.close()

# ---------- Utility helpers ----------
def generate_token():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=24))

def generate_vps_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def generate_ssh_password():
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choices(chars, k=14))

def has_admin_role(ctx):
    if isinstance(ctx, discord.Interaction):
        user_id = ctx.user.id
        try:
            roles = ctx.user.roles
        except:
            roles = []
    else:
        user_id = ctx.author.id
        roles = ctx.author.roles if hasattr(ctx.author, 'roles') else []

    if user_id in ADMIN_IDS:
        return True
    try:
        for role in roles:
            if getattr(role, 'id', None) == ADMIN_ROLE_ID:
                return True
    except:
        pass
    return False

def find_free_port(start=SSH_PORT_START, end=SSH_PORT_END):
    """Find a free TCP port on the host in range [start, end]"""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port available in the SSH port range")

# ---------- Bot class ----------
class LightplaysBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db = Database(DB_FILE)
        self.session = None
        self.docker_client = None
        self.system_stats = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_io': (0, 0),
            'last_updated': 0
        }

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
            self.loop.create_task(self.update_system_stats())
            self.loop.create_task(self.anti_miner_monitor())
            await self.reconnect_containers()
        except Exception as e:
            logger.error(f"Failed to init docker client: {e}")
            self.docker_client = None

    async def reconnect_containers(self):
        if not self.docker_client:
            return
        for token, vps in list(self.db.get_all_vps().items()):
            if vps['status'] == 'running':
                try:
                    container = self.docker_client.containers.get(vps['container_id'])
                    if container.status != 'running':
                        container.start()
                        logger.info(f"Started container for VPS {vps['vps_id']}")
                except docker.errors.NotFound:
                    logger.warning(f"Container {vps['container_id']} not found, removing DB entry")
                    self.db.remove_vps(token)
                except Exception as e:
                    logger.error(f"Error reconnecting {vps['vps_id']}: {e}")

    async def update_system_stats(self):
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                net_io = psutil.net_io_counters()
                self.system_stats = {
                    'cpu_usage': cpu_percent,
                    'memory_usage': mem.percent,
                    'memory_used': mem.used / (1024 ** 3),
                    'memory_total': mem.total / (1024 ** 3),
                    'disk_usage': disk.percent,
                    'disk_used': disk.used / (1024 ** 3),
                    'disk_total': disk.total / (1024 ** 3),
                    'network_sent': net_io.bytes_sent / (1024 ** 2),
                    'network_recv': net_io.bytes_recv / (1024 ** 2),
                    'last_updated': time.time()
                }
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
            await asyncio.sleep(30)

    async def anti_miner_monitor(self):
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                for token, vps in list(self.db.get_all_vps().items()):
                    if vps['status'] != 'running':
                        continue
                    try:
                        container = self.docker_client.containers.get(vps['container_id'])
                        if container.status != 'running':
                            continue
                        exec_result = container.exec_run("ps aux", stdout=True, stderr=True, demux=False)
                        output = exec_result.output.decode().lower() if exec_result and getattr(exec_result, 'output', None) else ""
                        for pattern in MINER_PATTERNS:
                            if pattern in output:
                                logger.warning(f"Mining detected in VPS {vps['vps_id']}. Suspending.")
                                try:
                                    container.stop()
                                except:
                                    pass
                                self.db.update_vps(token, {'status': 'suspended'})
                                try:
                                    owner = await self.fetch_user(int(vps['created_by']))
                                    await owner.send(f"⚠️ Your VPS {vps['vps_id']} has been suspended due to detected mining activity.")
                                except:
                                    pass
                                break
                    except Exception as e:
                        logger.error(f"Error checking vps {vps['vps_id']}: {e}")
            except Exception as e:
                logger.error(f"Anti miner loop error: {e}")
            await asyncio.sleep(300)

    async def close(self):
        await super().close()
        if self.session:
            await self.session.close()
        if self.docker_client:
            try:
                self.docker_client.close()
            except:
                pass
        self.db.close()

# instantiate bot
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = LightplaysBot(command_prefix='/', intents=intents, help_command=None)

# ---------- Low-level helpers using Docker ----------
async def build_custom_image_async(vps_id, username, root_password, user_password, base_image=DEFAULT_OS_IMAGE):
    """
    Build a custom image that runs sshd. Returns image tag.
    Building images is slow — prefer prebuilt images if possible.
    """
    temp_dir = f"temp_dockerfiles/{vps_id}"
    os.makedirs(temp_dir, exist_ok=True)
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        base_image=base_image,
        root_password=root_password,
        username=username,
        user_password=user_password
    )
    dockerfile_path = os.path.join(temp_dir, "Dockerfile")
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    image_tag = f"lightplays/{vps_id.lower()}:latest"
    proc = await asyncio.create_subprocess_exec(
        "docker", "build", "-t", image_tag, temp_dir,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    shutil.rmtree(temp_dir, ignore_errors=True)
    if proc.returncode != 0:
        raise Exception(stderr.decode() or "docker build failed")
    return image_tag

def allocate_port_and_run_container(image, vps_id, memory_gb, cpu_cores, disk_gb, volumespec=None, os_image=None):
    """
    Synchronous helper to run docker container with host port mapping for SSH
    Returns container object and host_ssh_port
    """
    memory_bytes = int(memory_gb * 1024 * 1024 * 1024)
    cpu_quota = int(cpu_cores * 100000)  # docker uses cpu_quota/cpu_period
    host_port = find_free_port()
    # set up docker run parameters
    ports = {'22/tcp': host_port}
    # create named volume for persistence
    volume_name = f"lightplays-{vps_id}-data"
    try:
        # create volume if not exists
        docker_client_local = docker.from_env()
        try:
            docker_client_local.volumes.get(volume_name)
        except docker.errors.NotFound:
            docker_client_local.volumes.create(name=volume_name)
        container = docker_client_local.containers.run(
            image,
            detach=True,
            hostname=f"lightplays-{vps_id}",
            mem_limit=memory_bytes,
            cpu_period=100000,
            cpu_quota=cpu_quota,
            network=DOCKER_NETWORK,
            ports=ports,
            volumes={volume_name: {'bind': '/data', 'mode': 'rw'}},
            restart_policy={"Name": "always"},
            tty=True,
            stdin_open=True
        )
        return container, host_port
    except Exception as e:
        raise

# ---------- Replace create_vps flow with dedicated-port mapping ----------
@bot.hybrid_command(name='create_vps', description='Create a new VPS (Admin only)')
@app_commands.describe(
    memory="Memory in GB",
    cpu="CPU cores",
    disk="Disk space in GB",
    owner="User who will own the VPS",
    os_image="OS image to use (must have SSHD)",
    use_custom_image="Build custom image with SSHD (slow)"
)
async def create_vps_command(ctx, memory: int, cpu: int, disk: int, owner: discord.Member,
                            os_image: str = DEFAULT_OS_IMAGE, use_custom_image: bool = False):
    """Create a dedicated VPS container exposed on its own host SSH port"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return

    if bot.db.is_user_banned(owner.id):
        await ctx.send("❌ This user is banned from creating VPS!", ephemeral=True)
        return

    if bot.db.get_user_vps_count(owner.id) >= bot.db.get_setting('max_vps_per_user', MAX_VPS_PER_USER):
        await ctx.send(f"❌ {owner.mention} already has the maximum number of VPS instances ({bot.db.get_setting('max_vps_per_user')})", ephemeral=True)
        return

    if not bot.docker_client:
        await ctx.send("❌ Docker is not available on the host.", ephemeral=True)
        return

    try:
        if memory < 0 or memory > 1024:
            await ctx.send("❌ Memory must be between 1 and 1024 GB", ephemeral=True)
            return
        if cpu < 1 or cpu > 64:
            await ctx.send("❌ CPU cores must be between 1 and 64", ephemeral=True)
            return
        if disk < 1 or disk > 5000:
            await ctx.send("❌ Disk must be between 1 and 5000 GB", ephemeral=True)
            return

        status_msg = await ctx.send("🚀 Creating dedicated VPS instance...")

        vps_id = generate_vps_id()
        username = (owner.name.lower().replace(" ", "_")[:16])
        root_password = generate_ssh_password()
        user_password = generate_ssh_password()
        token = generate_token()

        # Build custom image (optional)
        image_tag = os_image
        if use_custom_image:
            await status_msg.edit(content="🔨 Building custom SSH-enabled Docker image (this can take minutes)...")
            try:
                image_tag = await build_custom_image_async(vps_id, username, root_password, user_password, base_image=os_image)
            except Exception as e:
                logger.error(f"Custom image build failed: {e}")
                await status_msg.edit(content=f"❌ Failed to build custom image: {e}")
                return

        # Create container with host port mapping for SSH
        await status_msg.edit(content="⚙️ Starting container (SSH will be mapped to a host port)...")
        try:
            # Use the helper which chooses a free host port and runs the container
            container, host_ssh_port = allocate_port_and_run_container(
                image_tag, vps_id, memory, cpu, disk
            )
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            await status_msg.edit(content=f"❌ Failed to start container: {e}")
            return

        # Wait a few seconds for container to initialize
        await asyncio.sleep(5)

        # If we used a non-ssh-enabled image, try to install sshd inside container (best-effort)
        try:
            # Try to start sshd if it's present; otherwise try to install openssh-server (best-effort; can fail depending on image).
            exec_res = container.exec_run("which sshd || true")
            if not exec_res or exec_res.exit_code != 0:
                # Attempt package install (best-effort)
                logger.info("sshd not found in image; attempting apt-get install (best-effort).")
                # NOTE: this may fail on alpine images or minimal images
                install_cmd = container.exec_run("bash -lc \"apt-get update && apt-get install -y openssh-server || true\"", detach=False)
        except Exception as e:
            logger.warning(f"Attempt to ensure sshd failed: {e}")

        # Create user and set passwords inside container
        try:
            # Create user if not exists, set password, ensure sshd is running
            container.exec_run(f"useradd -m -s /bin/bash {username} || true")
            container.exec_run(f"bash -lc \"echo '{username}:{user_password}' | chpasswd\"")
            container.exec_run(f"bash -lc \"echo 'root:{root_password}' | chpasswd\"")
            # Try to start sshd inside container
            container.exec_run("bash -lc \"(which systemctl && systemctl restart ssh) || (/usr/sbin/sshd -D &>/dev/null & ) || true\"")
        except Exception as e:
            logger.warning(f"User/sshd setup may have failed: {e}")

        # Update DB with host_ssh_port
        vps_data = {
            "token": token,
            "vps_id": vps_id,
            "container_id": container.id,
            "host_ssh_port": host_ssh_port,
            "memory": memory,
            "cpu": cpu,
            "disk": disk,
            "username": username,
            "password": user_password,
            "root_password": root_password if use_custom_image else None,
            "created_by": str(owner.id),
            "created_at": str(datetime.datetime.utcnow()),
            "watermark": WATERMARK,
            "os_image": image_tag,
            "restart_count": 0,
            "last_restart": None,
            "status": "running",
            "use_custom_image": int(use_custom_image)
        }
        bot.db.add_vps(vps_data)

        # build response to owner (direct SSH)
        host_ip = os.getenv('HOST_PUBLIC_IP')  # set your server's public IP in .env
        if not host_ip:
            # fallback to discover IP (may return private ip behind NAT)
            try:
                host_ip = subprocess.check_output(["hostname", "-I"]).decode().split()[0]
            except Exception:
                host_ip = "your.host.ip"

        ssh_command = f"ssh {username}@{host_ip} -p {host_ssh_port}"
        try:
            embed = discord.Embed(title="🎉 Your Dedicated VPS is Ready", color=discord.Color.green())
            embed.add_field(name="VPS ID", value=vps_id, inline=True)
            embed.add_field(name="Memory", value=f"{memory}GB", inline=True)
            embed.add_field(name="CPU", value=f"{cpu} cores", inline=True)
            embed.add_field(name="Disk", value=f"{disk}GB", inline=True)
            embed.add_field(name="Username", value=username, inline=True)
            embed.add_field(name="Password", value=f"||{user_password}||", inline=False)
            if vps_data['root_password']:
                embed.add_field(name="Root Password", value=f"||{vps_data['root_password']}||", inline=False)
            embed.add_field(name="SSH Command", value=f"```{ssh_command}```", inline=False)
            embed.add_field(name="Note", value="This is a dedicated container with SSH exposed on a host port. Save the SSH command. If SSH doesn't connect immediately allow ~20s for the service to start.", inline=False)
            await owner.send(embed=embed)
            await status_msg.edit(content=f"✅ Dedicated VPS created for {owner.mention}. Connection details sent via DM.")
        except discord.Forbidden:
            await status_msg.edit(content=f"✅ VPS created but I couldn't DM {owner.mention}. SSH command: `{ssh_command}`")
    except Exception as e:
        logger.error(f"Error in create_vps_command: {e}")
        await ctx.send(f"❌ An error occurred while creating the VPS: {e}")
        # cleanup container on failure
        if 'container' in locals():
            try:
                container.stop()
                container.remove(force=True)
            except Exception as e2:
                logger.error(f"Cleanup failed: {e2}")

# ---------- Updated connect_vps: re-issue SSH details ----------
@bot.hybrid_command(name='connect_vps', description='Get SSH details for your VPS token')
@app_commands.describe(token="Access token for the VPS")
async def connect_vps(ctx, token: str):
    vps = bot.db.get_vps_by_token(token)
    if not vps:
        await ctx.send("❌ Invalid token!", ephemeral=True)
        return
    if str(ctx.author.id) != vps["created_by"] and not has_admin_role(ctx):
        await ctx.send("❌ You don't have permission to access this VPS!", ephemeral=True)
        return
    try:
        # ensure container is running
        try:
            container = bot.docker_client.containers.get(vps["container_id"])
            if container.status != "running":
                container.start()
                await asyncio.sleep(3)
        except Exception:
            await ctx.send("❌ VPS container not found on this host.", ephemeral=True)
            return

        host_ip = os.getenv('HOST_PUBLIC_IP')
        if not host_ip:
            try:
                host_ip = subprocess.check_output(["hostname", "-I"]).decode().split()[0]
            except:
                host_ip = "your.host.ip"

        ssh_command = f"ssh {vps['username']}@{host_ip} -p {vps['host_ssh_port']}"
        embed = discord.Embed(title=f"Connection for VPS {vps['vps_id']}", color=discord.Color.blue())
        embed.add_field(name="Username", value=vps['username'], inline=True)
        embed.add_field(name="Password", value=f"||{vps.get('password', 'Not set')}||", inline=True)
        embed.add_field(name="SSH Command", value=f"```{ssh_command}```", inline=False)
        await ctx.author.send(embed=embed)
        await ctx.send("✅ Connection details sent to your DMs.", ephemeral=True)
    except discord.Forbidden:
        await ctx.send("❌ I couldn't DM you. Enable DMs from server members.", ephemeral=True)
    except Exception as e:
        logger.error(f"connect_vps error: {e}")
        await ctx.send(f"❌ Error: {e}", ephemeral=True)

# ---------- The rest of your commands remain usable with dedicated-port model ----------
# list, admin_list_vps, delete_vps, vps_stats, change_ssh_password, admin_stats, system_info,
# cleanup_vps, suspend/unsuspend, edit_vps, transfer_vps, etc.
# For brevity, re-use original implementations but ensure they consult `host_ssh_port` and `container_id`.
# Here I provide a trimmed / fixed `delete_vps` and `list` to show adjustments:

@bot.hybrid_command(name='delete_vps', description='Delete a VPS instance (Admin only)')
@app_commands.describe(vps_id="ID of the VPS to delete")
async def delete_vps(ctx, vps_id: str):
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return
    try:
        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found!", ephemeral=True)
            return
        try:
            container = bot.docker_client.containers.get(vps["container_id"])
            container.stop(timeout=10)
            container.remove(force=True)
            # remove named volume (optional)
            vol_name = f"lightplays-{vps_id}-data"
            try:
                docker.from_env().volumes.get(vol_name).remove(force=True)
            except Exception:
                pass
            logger.info(f"Deleted container {vps['container_id']} for VPS {vps_id}")
        except docker.errors.NotFound:
            logger.warning("Container already removed")
        except Exception as e:
            logger.error(f"Error removing container: {e}")
        bot.db.remove_vps(token)
        await ctx.send(f"✅ VPS {vps_id} has been deleted.", ephemeral=True)
    except Exception as e:
        logger.error(f"delete_vps error: {e}")
        await ctx.send(f"❌ Error: {e}", ephemeral=True)

@bot.hybrid_command(name='list', description='List your VPS instances')
async def list_vps(ctx):
    try:
        user_vps = bot.db.get_user_vps(ctx.author.id)
        if not user_vps:
            await ctx.send("You don't have any VPS instances.", ephemeral=True)
            return
        embed = discord.Embed(title="Your Dedicated VPS Instances", color=discord.Color.blue())
        for vps in user_vps:
            status = vps.get('status', 'Unknown').capitalize()
            ssh_info = f"{vps.get('host_ssh_port','-')}"
            embed.add_field(
                name=f"VPS {vps['vps_id']}",
                value=f"Status: {status}\nMemory: {vps.get('memory','?')}GB\nCPU: {vps.get('cpu','?')} cores\nDisk: {vps.get('disk','?')}GB\nSSH Port: {ssh_info}\nOS: {vps.get('os_image', DEFAULT_OS_IMAGE)}",
                inline=False
            )
        await ctx.send(embed=embed, ephemeral=True)
    except Exception as e:
        logger.error(f"list_vps error: {e}")
        await ctx.send(f"❌ Error: {e}", ephemeral=True)

# ---------- Error handler ----------
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CheckFailure):
        await ctx.send("❌ You don't have permission to use this command!", ephemeral=True)
    elif isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ Command not found! Use `/help` to see available commands.", ephemeral=True)
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing required argument: {error.param.name}", ephemeral=True)
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(f"❌ An error occurred: {str(error)}", ephemeral=True)

# ---------- Startup ----------
if __name__ == "__main__":
    try:
        os.makedirs("temp_dockerfiles", exist_ok=True)
        os.makedirs("migrations", exist_ok=True)
        if not TOKEN:
            raise RuntimeError("DISCORD_TOKEN not set in environment (.env)")
        bot.run(TOKEN)
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        traceback.print_exc()
