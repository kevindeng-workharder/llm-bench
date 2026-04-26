"""Server lifecycle helper for benchmark runs.

Starts a VM-side server via SSH, opens a host→VM port-forward tunnel, waits
for /v1/models to respond, runs a callback, then tears everything down (kills
the server processes inside the VM AND force-kills the engine subprocess that
holds VRAM, then drops the tunnel).

Usage:
    from runner.server import VLLMServer
    with VLLMServer("vllm-qwen3-30b-awq-graph-tp1") as srv:
        srv.url   # http://localhost:8000
        # ... run benchmark
"""
from __future__ import annotations
import contextlib
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VM_HOST = "ubuntu@localhost"
DEFAULT_VM_PORT = "2222"
DEFAULT_LOCAL_PORT = 8000
DEFAULT_REMOTE_PORT = 8000


def _ssh(*args: str, vm_port: str = DEFAULT_VM_PORT,
         vm_host: str = DEFAULT_VM_HOST, check: bool = True,
         capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["ssh", "-p", vm_port, "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=5", vm_host, *args]
    return subprocess.run(cmd, check=check,
                          capture_output=capture, text=True)


def _wait_port(host: str, port: int, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def _wait_models_endpoint(url: str, deadline_s: float) -> bool:
    import httpx
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            r = httpx.get(f"{url}/v1/models", timeout=4)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


class RemoteServer:
    """Runs `bash <launch-script>` inside the VM, port-forwards to localhost.

    The launch script must:
      - exec the actual server (do NOT background it; we want PID linkage)
      - bind on 0.0.0.0:<port> inside the VM
      - source /home/ubuntu/vllm-serve/server-env.sh if needed
    """
    def __init__(self, name: str, launch_script_remote_path: str,
                 served_model_name: str,
                 vm_port: str = DEFAULT_VM_PORT,
                 vm_host: str = DEFAULT_VM_HOST,
                 local_port: int = DEFAULT_LOCAL_PORT,
                 remote_port: int = DEFAULT_REMOTE_PORT,
                 ready_timeout_s: int = 900,
                 log_path_remote: str = "/tmp/llm-bench.log"):
        self.name = name
        self.launch = launch_script_remote_path
        self.model = served_model_name
        self.vm_port = vm_port
        self.vm_host = vm_host
        self.local_port = local_port
        self.remote_port = remote_port
        self.ready_timeout_s = ready_timeout_s
        self.log_path = log_path_remote
        self.tunnel_proc: subprocess.Popen | None = None
        self.url = f"http://localhost:{self.local_port}"

    def __enter__(self):
        self.kill_remote()  # clean slate
        self._free_vram()
        self._start_remote()
        self._open_tunnel()
        self._wait_ready()
        return self

    def __exit__(self, exc_type, exc, tb):
        with contextlib.suppress(Exception):
            self._close_tunnel()
        with contextlib.suppress(Exception):
            self.kill_remote()
        with contextlib.suppress(Exception):
            self._free_vram()

    # --- Remote process lifecycle ---
    def _start_remote(self):
        # setsid+nohup so it survives the SSH session ending
        cmd = (
            f"setsid nohup bash {shlex.quote(self.launch)} "
            f"< /dev/null > {shlex.quote(self.log_path)} 2>&1 & disown; sleep 3; "
            f"pgrep -f vllm.entrypoints || pgrep -f llama-server || true"
        )
        print(f"[server:{self.name}] starting on VM…", file=sys.stderr)
        r = _ssh(cmd, vm_port=self.vm_port, vm_host=self.vm_host, capture=True, check=False)
        if r.returncode != 0:
            print(f"[server:{self.name}] start cmd exit={r.returncode} stderr={r.stderr.strip()}",
                  file=sys.stderr)

    def kill_remote(self):
        # Two passes: pkill the python/llama-server, then sudo-kill any stuck
        # VLLM::EngineCore subprocess that survives.
        _ssh("pkill -9 -f 'vllm.entrypoints'; pkill -9 -f 'llama-server'; true",
             vm_port=self.vm_port, vm_host=self.vm_host, check=False)
        time.sleep(1)
        _ssh("sudo kill -9 $(pgrep -f 'VLLM::Eng' 2>/dev/null) 2>/dev/null; true",
             vm_port=self.vm_port, vm_host=self.vm_host, check=False)
        time.sleep(2)

    def _free_vram(self):
        # Wait up to 30s for VRAM to drop below ~200 MB (post-cleanup baseline).
        deadline = time.time() + 30
        while time.time() < deadline:
            r = _ssh("cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null",
                     vm_port=self.vm_port, vm_host=self.vm_host,
                     capture=True, check=False)
            try:
                used = int(r.stdout.strip())
            except Exception:
                return
            if used < 200 * 1024 * 1024:
                return
            print(f"[server:{self.name}] waiting for VRAM release ({used/1e9:.1f}GB used)…",
                  file=sys.stderr)
            time.sleep(2)

    # --- Tunnel ---
    def _open_tunnel(self):
        # Force a fresh connection (close any pre-existing -L on this port).
        subprocess.run(["pkill", "-f", f"ssh.*-L *{self.local_port}:"],
                       check=False, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        cmd = ["ssh", "-p", self.vm_port, "-N",
               "-L", f"{self.local_port}:localhost:{self.remote_port}",
               "-o", "StrictHostKeyChecking=no",
               "-o", "ExitOnForwardFailure=yes",
               self.vm_host]
        self.tunnel_proc = subprocess.Popen(cmd)
        # Give the tunnel a moment to bind.
        if not _wait_port("127.0.0.1", self.local_port, 10):
            raise RuntimeError(f"tunnel for port {self.local_port} did not come up")

    def _close_tunnel(self):
        if self.tunnel_proc and self.tunnel_proc.poll() is None:
            self.tunnel_proc.terminate()
            try:
                self.tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tunnel_proc.kill()

    def _wait_ready(self):
        if not _wait_models_endpoint(self.url, self.ready_timeout_s):
            self.tail_log(60)
            raise RuntimeError(
                f"[server:{self.name}] not ready after {self.ready_timeout_s}s")
        print(f"[server:{self.name}] ready at {self.url}", file=sys.stderr)

    def tail_log(self, n: int = 30):
        r = _ssh(f"tail -n {n} {shlex.quote(self.log_path)}",
                 vm_port=self.vm_port, vm_host=self.vm_host,
                 capture=True, check=False)
        sys.stderr.write(r.stdout)
