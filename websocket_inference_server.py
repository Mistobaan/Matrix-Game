"""FastAPI module that serves Matrix-Game inference over websockets."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from utils.misc import set_seed
from websocket_server import MatrixGameInferenceService, ServiceOptions
from websocket_server.logging import RequestLoggingMiddleware
from websocket_server.models import SessionState


@dataclass
class ServerSettings:
    """CLI-friendly container for configuring the FastAPI server."""

    config_path: str = "configs/inference_yaml/inference_universal.yaml"
    checkpoint_path: str = ""
    output_folder: str = "outputs/"
    max_num_output_frames: int = 360
    pretrained_model_path: str = "."
    devices: List[str] = field(default_factory=lambda: ["cuda"])
    host: str = "0.0.0.0"
    port: int = 8765
    log_level: str = "INFO"
    seed: int = 0


class MatrixGameWebServer:
    """Bridges websocket events, session bookkeeping, and the inference engine."""

    def __init__(self, service: MatrixGameInferenceService):
        self.service = service
        self.logger = logging.getLogger("matrix_game.websocket")
        self._server_root = Path(__file__).resolve().parent
        self._sessions: Dict[WebSocket, SessionState] = {}
        self.demo_images = self._discover_demo_images()

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client = websocket.client
        self.logger.info(
            "WS connect peer=%s:%s",
            getattr(client, "host", "unknown"),
            getattr(client, "port", "unknown"),
        )

        session = None
        loop = asyncio.get_running_loop()

        await self._send_json(
            websocket,
            {
                "type": "ready",
                "modes": list(self.service.action_spaces.keys()),
                "max_frames": self.service.num_frames,
                "max_output_latents": self.service.workers[
                    0
                ].options.max_num_output_frames,
                "demo_images": self.demo_images,
            },
        )

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code"))

                data = message.get("text")
                if data is None and message.get("bytes") is not None:
                    try:
                        data = message["bytes"].decode("utf-8")
                    except UnicodeDecodeError:
                        await self._send_error(
                            websocket,
                            "invalid_encoding",
                            "Messages must be UTF-8 text.",
                        )
                        continue
                size = len(data.encode("utf-8")) if data else 0
                self.logger.info("WS recv size=%sB", size)

                if not data:
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    await self._send_error(
                        websocket, "invalid_json", "Failed to decode JSON payload."
                    )
                    continue

                msg_type = payload.get("type")
                if msg_type == "setup":
                    try:
                        session = await self._handle_setup(websocket, payload)
                    except Exception as exc:  # pylint: disable=broad-except
                        self.logger.exception("Failed to handle setup")
                        await self._send_error(websocket, "setup_failed", str(exc))
                elif msg_type == "action":
                    if not session:
                        await self._send_error(
                            websocket,
                            "no_session",
                            "Start a session before sending actions.",
                        )
                        continue
                    recorded = session.action_buffer.record_event(
                        payload.get("payload", {})
                    )
                    await self._send_json(
                        websocket,
                        {
                            "type": "action_recorded",
                            "session_id": session.session_id,
                            "frames_recorded": recorded,
                            "frames_remaining": session.num_frames - recorded,
                        },
                    )
                elif msg_type == "generate":
                    if not session:
                        await self._send_error(
                            websocket, "no_session", "Start a session before rendering."
                        )
                        continue
                    await self._handle_generate(websocket, session, loop)
                elif msg_type == "reset":
                    if not session:
                        await self._send_error(
                            websocket, "no_session", "Start a session before reset."
                        )
                        continue
                    await self._handle_reset(websocket, session, payload.get("payload"))
                elif msg_type == "ping":
                    await self._send_json(websocket, {"type": "pong"})
                else:
                    await self._send_error(
                        websocket, "unsupported", f"Unknown message type '{msg_type}'."
                    )
        except WebSocketDisconnect:
            self.logger.info("WS disconnect")
        finally:
            if session:
                self._sessions.pop(websocket, None)

    async def _handle_setup(self, websocket: WebSocket, message: Dict[str, Any]):
        image_path = message.get("image_path")
        if not image_path:
            raise ValueError("`image_path` is required in setup messages.")
        mode = message.get("mode")
        session_id = message.get("session_id")

        session = await self.service.prepare_session(image_path, mode, session_id)
        self._sessions[websocket] = session
        await self._send_json(
            websocket,
            {
                "type": "session_started",
                "session_id": session.session_id,
                "mode": session.mode,
                "num_frames": session.num_frames,
            },
        )
        return session

    async def _handle_generate(
        self,
        websocket: WebSocket,
        session: SessionState,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        def frame_callback(
            index: int,
            total: int,
            frame_b64: str,
            width: int,
            height: int,
            format_: str,
        ) -> None:
            payload = {
                "type": "frame",
                "session_id": session.session_id,
                "frame_index": index,
                "total_frames": total,
                "width": width,
                "height": height,
                "format": format_,
                "data": frame_b64,
            }
            future = asyncio.run_coroutine_threadsafe(
                self._send_json(websocket, payload), loop
            )
            self._monitor_background(future, "frame_stream")

        try:
            result = await self.service.generate(session, frame_callback=frame_callback)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Generation failed")
            await self._send_error(websocket, "generate_failed", str(exc))
            return

        await self._send_json(
            websocket,
            {
                "type": "generation_complete",
                "session_id": session.session_id,
                **result,
            },
        )

    async def _handle_reset(
        self,
        websocket: WebSocket,
        session: SessionState,
        payload: Optional[Dict[str, Any]],
    ) -> None:
        session.reset_actions()
        if payload and payload.get("clear_session"):
            image_override = payload.get("image_path")
            mode_override = payload.get("mode")
            try:
                session = await self.service.prepare_session(
                    image_override or session.image_path,
                    mode_override or session.mode,
                    session.session_id,
                )
                self._sessions[websocket] = session
                await self._send_json(
                    websocket,
                    {
                        "type": "setup",
                        "image_path": image_override or session.image_path,
                        "mode": session.mode,
                        "session_id": session.session_id,
                    },
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("Failed to reset session")
                await self._send_error(websocket, "reset_failed", str(exc))
        else:
            await self._send_json(
                websocket,
                {
                    "type": "session_reset",
                    "session_id": session.session_id,
                    "frames_recorded": 0,
                    "frames_remaining": session.num_frames,
                },
            )

    async def _send_json(self, websocket: WebSocket, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload)
        await websocket.send_text(message)
        payload_type = payload.get("type")
        size = len(message.encode("utf-8"))
        self.logger.info("WS send type=%s size=%sB", payload_type, size)

    async def _send_error(self, websocket: WebSocket, code: str, message: str) -> None:
        await self._send_json(
            websocket, {"type": "error", "error": {"code": code, "message": message}}
        )

    def _discover_demo_images(self) -> Dict[str, List[str]]:
        root = self._server_root / "demo_images"
        mapping: Dict[str, List[str]] = {}
        for mode in self.service.action_spaces:
            directory = root / self._mode_to_demo_dir(mode)
            if not directory.exists():
                continue
            files: List[Path] = []
            for pattern in ("*.png", "*.jpg", "*.jpeg"):
                files.extend(directory.glob(pattern))
            files = sorted(files)
            if not files:
                continue
            mapping[mode] = [self._relative_path(path) for path in files]
        return mapping

    def _mode_to_demo_dir(self, mode: str) -> str:
        override = {
            "templerun": "temple_run",
        }
        return override.get(mode, mode)

    def _relative_path(self, path: Path) -> str:
        for base in (Path.cwd(), self._server_root):
            try:
                return str(path.relative_to(base))
            except ValueError:
                continue
        return str(path)

    def _monitor_background(self, future: "asyncio.Future[Any]", context: str) -> None:
        def _done_callback(fut: "asyncio.Future[Any]") -> None:
            if fut.cancelled():
                return
            try:
                fut.result()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error(
                    "Background task %s failed: %s", context, exc, exc_info=exc
                )

        future.add_done_callback(_done_callback)


def create_app(settings: Optional[ServerSettings] = None) -> FastAPI:
    settings = settings or ServerSettings()

    set_seed(settings.seed)
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        ),
        Middleware(
            RequestLoggingMiddleware, logger=logging.getLogger("matrix_game.web")
        ),
    ]
    app = FastAPI(middleware=middleware)

    service = MatrixGameInferenceService(
        ServiceOptions(
            config_path=settings.config_path,
            checkpoint_path=settings.checkpoint_path,
            output_folder=settings.output_folder,
            max_num_output_frames=settings.max_num_output_frames,
            pretrained_model_path=settings.pretrained_model_path,
            devices=settings.devices,
        )
    )
    server = MatrixGameWebServer(service)

    app.state.inference_service = service
    app.state.web_server = server

    @app.get("/", response_class=FileResponse)
    async def serve_index() -> FileResponse:
        index_path = Path(__file__).resolve().with_name("index.html")
        return FileResponse(index_path)

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/modes")
    async def list_modes() -> Dict[str, Any]:
        return {
            "default_mode": service.default_mode,
            "modes": list(service.action_spaces.keys()),
        }

    @app.get("/demo-images")
    async def demo_images() -> Dict[str, List[str]]:
        return server.demo_images

    @app.get("/demo_images/{resource_path:path}", response_class=FileResponse)
    async def fetch_demo_asset(resource_path: str) -> FileResponse:
        demo_root = Path(__file__).resolve().parent / "demo_images"
        candidate = (demo_root / resource_path).resolve()
        if not candidate.is_file() or demo_root not in candidate.parents:
            raise HTTPException(status_code=404, detail="Demo asset not found.")
        return FileResponse(candidate)

    @app.websocket("/ws")
    async def websocket_route(websocket: WebSocket) -> None:
        await server.websocket_endpoint(websocket)

    return app


def parse_args() -> ServerSettings:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=ServerSettings.config_path,
        help="Path to OmegaConf config file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=ServerSettings.checkpoint_path,
        help="Optional diffusion checkpoint in safetensors format.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=ServerSettings.output_folder,
        help="Output directory for rendered videos.",
    )
    parser.add_argument(
        "--max_num_output_frames",
        type=int,
        default=ServerSettings.max_num_output_frames,
        help="Max number of latent frames (matches original CLI).",
    )
    parser.add_argument(
        "--seed", type=int, default=ServerSettings.seed, help="Random seed."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=ServerSettings.pretrained_model_path,
        help="Directory containing Wan VAE weights.",
    )
    parser.add_argument(
        "--host", type=str, default=ServerSettings.host, help="HTTP bind host."
    )
    parser.add_argument(
        "--port", type=int, default=ServerSettings.port, help="HTTP bind port."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=ServerSettings.log_level,
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda",
        help="Comma separated list of devices, e.g. 'cuda:0,cuda:1'.",
    )
    args = parser.parse_args()

    devices = [dev.strip() for dev in args.devices.split(",") if dev.strip()]
    if not devices:
        devices = ["cuda"]

    return ServerSettings(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_folder=args.output_folder,
        max_num_output_frames=args.max_num_output_frames,
        pretrained_model_path=args.pretrained_model_path,
        devices=devices,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        seed=args.seed,
    )


def main() -> None:
    settings = parse_args()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ != "__main__":
    app = create_app()


if __name__ == "__main__":
    main()
