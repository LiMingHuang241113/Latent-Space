# src/transport.py
import asyncio
import websockets

class ObservationServer:
    """
    Minimal WebSocket broadcast server (compatible with websockets >= 11).
    Clients connect to ws://<host>:<port><path>
    """
    def __init__(self, cfg):
        net = cfg.get("network", {})
        self.host = net.get("host", "127.0.0.1")
        self.port = int(net.get("port", 8765))
        self.path = net.get("path", "/observations")
        self.clients = set()
        self._server = None

    async def start(self):
        async def handler(ws):
            try:
                # Relaxed path rule: allow "/" and the configured path
                client_path = getattr(ws, "path", "/")
                if client_path not in (self.path, "/"):
                    print(f"[WS] rejecting client with path {client_path!r}, expected {self.path!r}")
                    await ws.close(code=1008, reason=f"Wrong path {client_path}")
                    return

                self.clients.add(ws)
                print(f"[WS] client connected at {client_path}, total {len(self.clients)}")
                try:
                    # We don't expect client messages; keep connection alive.
                    async for _ in ws:
                        pass
                finally:
                    self.clients.discard(ws)
                    print(f"[WS] client disconnected, remaining {len(self.clients)}")

            except Exception as e:
                print(f"[WS] internal error: {e}")
                try:
                    await ws.close(code=1011, reason="Server error")
                except Exception:
                    pass

        # Start server
        self._server = await websockets.serve(
            handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**20,
        )
        print(f"[WS] server listening on ws://{self.host}:{self.port}{self.path}")

    async def broadcast(self, msg: str):
        """Send a JSON string to all connected clients; prune dead sockets."""
        if not self.clients:
            return
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
