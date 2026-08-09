import subprocess
import sys
import time
import os
import webbrowser
import threading
import shutil
from pathlib import Path

def find_npm():
    """Find npm executable, handling Windows npm.cmd quirk."""
    npm = shutil.which("npm")
    if npm:
        return npm
    # On Windows, npm is npm.cmd - find it next to node.exe
    node = shutil.which("node")
    if node:
        npm_cmd = Path(node).parent / "npm.cmd"
        if npm_cmd.exists():
            return str(npm_cmd)
    return None

BACKEND_PORT = 8000
FRONTEND_PORT = 5173

BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"

browser_opened = False
browser_lock = threading.Lock()


def log(msg):
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def check_port(port):
    import socket
    # Try IPv4 first
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", port))
    sock.close()
    if result == 0:
        return True
    # Try IPv6
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        result = sock.connect_ex(("::1", port))
        sock.close()
        return result == 0
    except:
        return False


def wait_for_url(url, timeout=60, check_interval=0.5):
    import urllib.request
    import urllib.error

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=2)
            if response.status == 200:
                return True
        except (urllib.error.URLError, Exception):
            pass

        time.sleep(check_interval)

    return False


def open_browser_once():
    global browser_opened

    with browser_lock:
        if not browser_opened:
            log("Opening browser...")
            webbrowser.open(FRONTEND_URL)
            browser_opened = True


def copy_video_to_public():
    video_sources = [
        DATA_DIR / "assets.mp4",
        DATA_DIR / "asset.mp4",
        DATA_DIR / "flood_demo.mp4",
    ]
    video_dest = FRONTEND_DIR / "public" / "asset.mp4"

    video_source = None
    for src in video_sources:
        if src.exists():
            video_source = src
            break

    if video_source:
        video_dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(video_source, video_dest)
        log(f"Copied video to public directory: {video_source.name}")
    else:
        log("WARNING: No video source found in data directory")
        log(f"Looking in: {DATA_DIR}")
        video_dest.parent.mkdir(parents=True, exist_ok=True)


def start_backend():
    log("Starting backend server...")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    log(f"Waiting for backend at {BACKEND_URL}...")

    if wait_for_url(BACKEND_URL, timeout=30):
        log("Backend is ready!")
    else:
        log("WARNING: Backend may not be ready")

    return process


def start_frontend():
    log("Starting frontend server...")

    npm_cmd = find_npm()
    if not npm_cmd:
        log("ERROR: npm not found in PATH. Is Node.js installed?")
        return None

    if not (FRONTEND_DIR / "node_modules").exists():
        log("Installing frontend dependencies...")
        subprocess.run([npm_cmd, "install"], check=True, cwd=str(FRONTEND_DIR))

    process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    log(f"Waiting for frontend at {FRONTEND_URL}...")

    if wait_for_url(FRONTEND_URL, timeout=60):
        log("Frontend is ready!")
        open_browser_once()
    else:
        log("WARNING: Frontend may not be ready")

    return process


def main():
    print("=" * 60)
    print("HYDROSIGNAL - FLOOD EARLY WARNING SYSTEM")
    print("=" * 60)
    print()

    copy_video_to_public()

    backend_process = start_backend()
    frontend_process = start_frontend()

    print()
    print("=" * 60)
    print("SYSTEM RUNNING")
    print("=" * 60)
    print(f"Backend:  {BACKEND_URL}")
    print(f"Frontend: {FRONTEND_URL}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        # Simple wait loop - just wait for keyboard interrupt
        # Don't check process status since Popen.poll() can be unreliable
        while True:
            time.sleep(1)

            # Check if port is still open
            if not check_port(BACKEND_PORT):
                log("WARNING: Backend port not responding")
                break
            if not check_port(FRONTEND_PORT):
                log("WARNING: Frontend port not responding")
                break

    except KeyboardInterrupt:
        print()
        log("Shutting down...")

    log("Stopping processes...")

    if backend_process:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except:
            backend_process.kill()

    if frontend_process:
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except:
            frontend_process.kill()

    log("Done")


if __name__ == "__main__":
    main()
