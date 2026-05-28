from __future__ import annotations

import json
import socket
from typing import Any


def send_json_request(host: str, port: int, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    message = (json.dumps(payload) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(message)
        chunks = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw_response = b"".join(chunks).split(b"\n", 1)[0]
    if not raw_response:
        raise RuntimeError("Empty response from prediction socket server.")
    response = json.loads(raw_response.decode("utf-8"))
    if not response.get("ok", False):
        raise RuntimeError(response.get("error", "Prediction socket server returned an error."))
    return response
