from __future__ import annotations

import argparse
import json
import socket
import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from inference import DEFAULT_MODEL_DIR, PredictionService
from socket_protocol import send_json_request


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mRNA stability inference server.")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=17888)
    parser.add_argument("--socket-host", default="127.0.0.1")
    parser.add_argument("--socket-port", type=int, default=16888)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--debug", action="store_true")
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


def create_app(socket_host: str, socket_port: int) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )

    @app.get("/")
    def root():
        return render_template("index.html", result=None, error=None)

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.get("/2026Project/")
    def project_home():
        return render_template("index.html", result=None, error=None)

    @app.post("/2026Project/predict")
    def predict():
        payload = {
            "action": "predict",
            "transcript_id": request.form.get("transcript_id", "query"),
            "5UTRseq": request.form.get("utr5", ""),
            "CDSseq": request.form.get("cds", ""),
            "3UTRseq": request.form.get("utr3", ""),
            "threshold": request.form.get("threshold", "0.5"),
        }
        try:
            response = send_json_request(socket_host, socket_port, payload)
            return render_template("index.html", result=response["prediction"], error=None)
        except Exception as exc:
            return render_template("index.html", result=None, error=str(exc)), 400

    @app.post("/2026Project/api/predict")
    def api_predict():
        payload = request.get_json(silent=True) or {}
        payload["action"] = "predict"
        response = send_json_request(socket_host, socket_port, payload)
        return jsonify(response)

    @app.get("/2026Project/api/health")
    def health():
        response = send_json_request(socket_host, socket_port, {"action": "health"}, timeout=10.0)
        return jsonify(response)

    return app


def main() -> None:
    args = parse_args()
    service = PredictionService(args.model_dir)
    socket_server = PredictionSocketServer(args.socket_host, args.socket_port, service)
    socket_thread = threading.Thread(target=socket_server.serve_forever, daemon=True)
    socket_thread.start()

    app = create_app(args.socket_host, args.socket_port)
    print(f"HTTP server available at http://{args.http_host}:{args.http_port}/2026Project/", flush=True)
    app.run(host=args.http_host, port=args.http_port, debug=args.debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
