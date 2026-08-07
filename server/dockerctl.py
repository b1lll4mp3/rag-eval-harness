"""Minimal Docker Engine API client over /var/run/docker.sock. Stdlib only.

No delete endpoints on purpose: this module can start, stop, inspect and
exec, and nothing else.
"""
import json
import socket
import time

SOCK = "/var/run/docker.sock"
API = "/v1.41"


def build_request(method, path, body=None):
    payload = b""
    # Connection: close is load-bearing, since the reader drains until the peer
    # closes, and Docker's default keep-alive would block recv until timeout.
    headers = [f"{method} {API}{path} HTTP/1.1", "Host: docker",
               "Connection: close"]
    if body is not None:
        payload = json.dumps(body).encode()
        headers += ["Content-Type: application/json",
                    f"Content-Length: {len(payload)}"]
    else:
        headers += ["Content-Length: 0"]
    return ("\r\n".join(headers) + "\r\n\r\n").encode() + payload


def docker_request(method, path, body=None, timeout=60):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(SOCK)
        s.sendall(build_request(method, path, body))
        raw = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            raw += chunk
    head, _, rest = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    if b"Transfer-Encoding: chunked" in head:
        out, buf = b"", rest
        while buf:
            size_line, _, buf = buf.partition(b"\r\n")
            size = int(size_line, 16)
            if size == 0:
                break
            out, buf = out + buf[:size], buf[size + 2:]
        rest = out
    text = rest.decode("utf-8", "replace").strip()
    try:
        return status, json.loads(text) if text else {}
    except json.JSONDecodeError:
        return status, text


def container_running(name):
    status, data = docker_request("GET", f"/containers/{name}/json")
    return status == 200 and data.get("State", {}).get("Running", False)


def start(name):
    status, _ = docker_request("POST", f"/containers/{name}/start")
    if status not in (204, 304):
        raise RuntimeError(f"start {name} -> HTTP {status}")


def stop(name):
    status, _ = docker_request("POST", f"/containers/{name}/stop?t=30", timeout=90)
    if status not in (204, 304):
        raise RuntimeError(f"stop {name} -> HTTP {status}")


def exec_in(name, cmd):
    status, data = docker_request("POST", f"/containers/{name}/exec",
                                  {"Cmd": cmd, "AttachStdout": True, "AttachStderr": True})
    if status != 201:
        raise RuntimeError(f"exec create in {name} -> HTTP {status}")
    exec_id = data["Id"]
    status, out = docker_request("POST", f"/exec/{exec_id}/start",
                                 {"Detach": False, "Tty": True})
    if status != 200:
        raise RuntimeError(f"exec start -> HTTP {status}")
    return out if isinstance(out, str) else json.dumps(out)


def restart_with_env(name, env_overrides):
    """Recreate a container with modified env. Rename-only strategy: no delete
    paths, no fixed alias names, so repeat swaps can never collide: each swap
    parks the old container as {name}-old-<unix-ts>. Stale parked containers
    accumulate stopped and harmless; prune them manually if they bother you.

    Failure behavior: the live container is renamed BEFORE it is stopped, so a
    rename refusal leaves the service untouched. If create/start of the new
    container fails, the broken one (if any) is shoved aside to
    {name}-broken-<ts>, the old container is renamed back and restarted."""
    status, cfg = docker_request("GET", f"/containers/{name}/json")
    if status != 200:
        raise RuntimeError(f"inspect {name} -> HTTP {status}")
    env = dict(e.split("=", 1) for e in cfg["Config"]["Env"] if "=" in e)
    env.update(env_overrides)
    create_body = {
        "Image": cfg["Config"]["Image"],
        "Env": [f"{k}={v}" for k, v in env.items()],
        "HostConfig": cfg["HostConfig"],
        "ExposedPorts": cfg["Config"].get("ExposedPorts", {}),
    }
    ts = int(time.time())
    old = f"{name}-old-{ts}"
    status, _ = docker_request("POST", f"/containers/{name}/rename?name={old}")
    if status != 204:
        # nothing has been touched yet, so the live container keeps running
        raise RuntimeError(f"rename {name} -> {old} refused: HTTP {status}")
    try:
        stop(old)
        status, data = docker_request("POST", f"/containers/create?name={name}", create_body)
        if status != 201:
            raise RuntimeError(f"create {name} -> HTTP {status}: {data}")
        start(name)
    except Exception:
        # a failed create may still have claimed the name, so shove it aside
        docker_request("POST", f"/containers/{name}/rename?name={name}-broken-{ts}")
        status, _ = docker_request("POST", f"/containers/{old}/rename?name={name}")
        if status != 204:
            raise RuntimeError(
                f"ROLLBACK FAILED: old container stranded as {old} (HTTP {status})")
        start(name)
        raise
    return True
