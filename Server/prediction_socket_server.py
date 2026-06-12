from __future__ import annotations

import argparse
import json
import socket
import threading
from pathlib import Path
from typing import Any

from inference import DEFAULT_MODEL_DIR, PredictionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mRNA stability prediction socket server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=16888)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


class PredictionSocketServer:
    def __init__(self, host: str, port: int, service: PredictionService):
        self.host = host
        self.port = port
        self.service = service
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(16)
            server.settimeout(1.0)
            print(f"Prediction socket server listening on {self.host}:{self.port}", flush=True)
            while not self._stop.is_set():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_client, args=(connection, address), daemon=True).start()

    def _handle_client(self, connection: socket.socket, address: tuple[str, int]) -> None:
        with connection:
            try:
                data = b""
                while b"\n" not in data:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    return
                payload = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
                response = self._dispatch(payload)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            connection.sendall((json.dumps(response) + "\n").encode("utf-8"))

    def _dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action", "predict")
        if action == "health":
            return {"ok": True, "metadata": self.service.metadata()}
        if action == "predict":
            return {"ok": True, "prediction": self.service.predict(payload)}
        raise ValueError(f"Unsupported socket action: {action}")


def main() -> None:
    args = parse_args()
    service = PredictionService(args.model_dir)
    PredictionSocketServer(args.host, args.port, service).serve_forever()


if __name__ == "__main__":
    main()
