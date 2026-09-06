"""Start the local kitchen with its private cloud-QA configuration and local PostgreSQL.

Run from any directory: python scripts/cooperative_kitchen/start_local.py
The separate deployment.env keeps its Neon URL for Render; this launcher does
not rewrite it. No credentials are printed or placed in command-line arguments.
"""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "output/cooperative_kitchen/private/deployment.env"
PYTHON = ROOT / "output/cooperative_kitchen/.cpu-deployment-venv/bin/python"
PG_BIN = Path("/opt/homebrew/opt/postgresql@17/bin")
PG_DATA = ROOT / "output/cooperative_kitchen/postgres"
PG_SOCKET = Path("/tmp/policylens-kitchen-pg")


def main():
    if not PRIVATE.is_file() or PRIVATE.stat().st_mode & 0o077:
        raise SystemExit("Private deployment.env is missing or must have mode 0600.")
    if not PYTHON.is_file():
        raise SystemExit("The local CPU environment is missing; see the deployment report.")
    env = os.environ.copy()
    for line in PRIVATE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("\"'")
    if not env.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is missing from the private configuration.")
    if not PG_DATA.is_dir() or not (PG_BIN / "pg_ctl").is_file():
        raise SystemExit("The existing local PostgreSQL 17 installation is missing.")
    ready = subprocess.run([str(PG_BIN / "pg_isready"), "-h", str(PG_SOCKET), "-p", "55432"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ready.returncode:
        PG_SOCKET.mkdir(mode=0o700, exist_ok=True)
        subprocess.run([str(PG_BIN / "pg_ctl"), "-D", str(PG_DATA), "-l", str(PG_DATA / "server.log"),
                        "-o", f"-k {PG_SOCKET} -p 55432 -c listen_addresses=''", "start"], check=True)
    env.update(DATABASE_URL=f"postgresql+psycopg:///kitchen_development?host={PG_SOCKET}&port=55432",
               KITCHEN_NAMESPACE="development", KITCHEN_SECURE_COOKIE="0", KITCHEN_QA_WORKERS="2",
               KITCHEN_FREEPLAY_QA="1",
               KITCHEN_OUTPUT=str(ROOT / "output/cooperative_kitchen/v3-id-pilot"))
    os.chdir(ROOT)
    os.execve(str(PYTHON), [str(PYTHON), "-m", "ui.cooperative_kitchen_server",
                          "--host", "127.0.0.1", "--port", "8003", *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
