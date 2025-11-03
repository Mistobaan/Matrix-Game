import argparse
import asyncio
import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers.utils import load_image
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.transforms import v2
from PIL import Image

from pipeline import CausalInferenceStreamingPipeline
from safetensors.torch import load_file
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.misc import set_seed
from utils.visualize import process_video
from utils.wan_wrapper import WanDiffusionWrapper
from wan.vae.wanx_vae import get_wanx_vae_wrapper

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
    from websockets.server import WebSocketServerProtocol
except ImportError as exc:
    raise ImportError(
        "Install the `websockets` package to use the websocket inference server "
        "(pip install websockets)."
    ) from exc


@dataclass(frozen=True)
class ActionSpace:
    mode: str
    keyboard_map: Dict[str, int]
    default_keys: Tuple[str, ...] = ()
    enable_mouse: bool = True

    @property
    def keyboard_dim(self) -> int:
        return len(self.keyboard_map)


class ActionBuffer:
    """Accumulates incremental keyboard/mouse events into dense frame tensors."""

    def __init__(self, space: ActionSpace, num_frames: int):
        self.space = space
        self.num_frames = num_frames
        self.keyboard = torch.zeros((num_frames, space.keyboard_dim), dtype=torch.float32)
        self.mouse = torch.zeros((num_frames, 2), dtype=torch.float32) if space.enable_mouse else None
        self._cursor = 0
        self._last_keys: Tuple[str, ...] = space.default_keys
        self._last_mouse_vec = torch.zeros(2, dtype=torch.float32) if space.enable_mouse else None

    @property
    def frames_recorded(self) -> int:
        return self._cursor

    def is_full(self) -> bool:
        return self._cursor >= self.num_frames

    def record_event(self, payload: Dict[str, Any]) -> int:
        """Ingest one websocket action payload."""
        if self.is_full():
            return self._cursor

        repeat = max(int(payload.get("repeat", 1)), 1)

        keyboard_state = payload.get("keyboard", {}).get("state")
        if keyboard_state is not None:
            if isinstance(keyboard_state, str):
                keyboard_state = [keyboard_state]
            self._last_keys = tuple(keyboard_state)

        mouse_state = payload.get("mouse")
        if self.space.enable_mouse and mouse_state is not None:
            self._last_mouse_vec = self._mouse_vector(mouse_state)

        for _ in range(repeat):
            if self._cursor >= self.num_frames:
                break
            self.keyboard[self._cursor] = self._keyboard_row(self._last_keys)
            if self.space.enable_mouse and self.mouse is not None:
                mouse_vec = self._last_mouse_vec if self._last_mouse_vec is not None else torch.zeros(2, dtype=torch.float32)
                self.mouse[self._cursor] = mouse_vec
            self._cursor += 1

        return self._cursor

    def export(self, device: torch.device, dtype: torch.dtype):
        """Finalize and return GPU and CPU tensors for inference and overlays."""
        self._finalize()
        keyboard_cpu = self.keyboard.clone()
        keyboard = keyboard_cpu.unsqueeze(0).to(device=device, dtype=dtype)
        mouse_cpu = None
        mouse = None
        if self.space.enable_mouse and self.mouse is not None:
            mouse_cpu = self.mouse.clone()
            mouse = mouse_cpu.unsqueeze(0).to(device=device, dtype=dtype)
        return keyboard, mouse, keyboard_cpu, mouse_cpu

    def reset(self) -> None:
        self.keyboard.zero_()
        if self.mouse is not None:
            self.mouse.zero_()
        self._cursor = 0
        self._last_keys = self.space.default_keys
        self._last_mouse_vec = torch.zeros(2, dtype=torch.float32) if self.space.enable_mouse else None

    def _keyboard_row(self, keys: Optional[Tuple[str, ...]]) -> torch.Tensor:
        row = torch.zeros(self.space.keyboard_dim, dtype=torch.float32)
        if keys:
            for key in keys:
                idx = self.space.keyboard_map.get(key)
                if idx is not None:
                    row[idx] = 1.0
        else:
            for key in self.space.default_keys:
                idx = self.space.keyboard_map.get(key)
                if idx is not None:
                    row[idx] = 1.0
        return row

    def _mouse_vector(self, mouse_state: Dict[str, Any]) -> torch.Tensor:
        dx = float(mouse_state.get("dx", 0.0))
        dy = float(mouse_state.get("dy", 0.0))
        return torch.tensor([dy, dx], dtype=torch.float32)

    def _finalize(self) -> None:
        if self._cursor == 0:
            base_row = self._keyboard_row(self.space.default_keys)
            for idx in range(self.num_frames):
                self.keyboard[idx] = base_row
            if self.space.enable_mouse and self.mouse is not None:
                fill = self._last_mouse_vec if self._last_mouse_vec is not None else torch.zeros(2, dtype=torch.float32)
                for idx in range(self.num_frames):
                    self.mouse[idx] = fill
            self._cursor = self.num_frames
            return

        last_key_row = self.keyboard[self._cursor - 1].clone()
        last_mouse_row = self.mouse[self._cursor - 1].clone() if self.space.enable_mouse and self.mouse is not None else None
        for idx in range(self._cursor, self.num_frames):
            self.keyboard[idx] = last_key_row
            if last_mouse_row is not None:
                self.mouse[idx] = last_mouse_row
        self._cursor = self.num_frames


@dataclass
class SessionState:
    session_id: str
    mode: str
    image_path: str
    cond_concat: torch.Tensor
    visual_context: torch.Tensor
    action_space: ActionSpace
    action_buffer: ActionBuffer
    num_frames: int
    output_stem: Path

    def reset_actions(self) -> None:
        self.action_buffer.reset()


class WebsocketInteractiveInference:
    """Owns model weights and performs GPU-accelerated generation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.num_frames = (self.args.max_num_output_frames - 1) * 4 + 1
        self.output_dir = Path(self.args.output_folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mouse_icon = Path(__file__).resolve().parent / "assets" / "images" / "mouse.png"
        self.generation_lock = asyncio.Lock()

        self.action_spaces = {
            "universal": ActionSpace(
                mode="universal",
                keyboard_map={"forward": 0, "back": 1, "left": 2, "right": 3},
                enable_mouse=True,
            ),
            "gta_drive": ActionSpace(
                mode="gta_drive",
                keyboard_map={"forward": 0, "back": 1},
                enable_mouse=True,
            ),
            "templerun": ActionSpace(
                mode="templerun",
                keyboard_map={
                    "nomove": 0,
                    "jump": 1,
                    "slide": 2,
                    "turnleft": 3,
                    "turnright": 4,
                    "leftside": 5,
                    "rightside": 6,
                },
                default_keys=("nomove",),
                enable_mouse=False,
            ),
        }

    def _init_config(self) -> None:
        self.config = OmegaConf.load(self.args.config_path)
        self.default_mode = self.config.pop("mode") if "mode" in self.config else "universal"

    def _init_models(self) -> None:
        generator = WanDiffusionWrapper(**getattr(self.config, "model_kwargs", {}), is_causal=True)
        vae_decoder = VAEDecoderWrapper()
        vae_state = torch.load(
            os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"),
            map_location="cpu",
        )
        decoder_state = {key: value for key, value in vae_state.items() if "decoder." in key or "conv2" in key}
        vae_decoder.load_state_dict(decoder_state)
        vae_decoder.to(self.device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        pipeline = CausalInferenceStreamingPipeline(self.config, generator=generator, vae_decoder=vae_decoder)
        if self.args.checkpoint_path:
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def prepare_session(self, image_path: str, mode: Optional[str], session_id: Optional[str] = None) -> SessionState:
        target_mode = mode or self.default_mode
        if target_mode not in self.action_spaces:
            raise ValueError(f"Unsupported mode '{target_mode}'. Valid modes: {list(self.action_spaces)}")

        conds = self._prepare_conditionals(image_path)
        session_uuid = session_id or uuid.uuid4().hex[:8]
        action_space = self.action_spaces[target_mode]
        buffer = ActionBuffer(action_space, self.num_frames)
        output_stem = (self.output_dir / session_uuid).with_suffix("")

        return SessionState(
            session_id=session_uuid,
            mode=target_mode,
            image_path=image_path,
            cond_concat=conds["cond_concat"],
            visual_context=conds["visual_context"],
            action_space=action_space,
            action_buffer=buffer,
            num_frames=self.num_frames,
            output_stem=output_stem,
        )

    async def generate(
        self,
        session: SessionState,
        frame_callback: Optional[Callable[[int, int, str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        async with self.generation_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._generate_sync, session, frame_callback, loop)

    def _prepare_conditionals(self, image_path: str) -> Dict[str, torch.Tensor]:
        tiler_kwargs = {
            "tiled": True,
            "tile_size": [44, 80],
            "tile_stride": [23, 38],
        }

        image = load_image(image_path)
        image = self._resize_crop(image, 352, 640)
        with torch.no_grad():
            image_tensor = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
            padding = torch.zeros_like(image_tensor).repeat(
                1, 1, 4 * (self.args.max_num_output_frames - 1), 1, 1
            )
            img_cond = torch.cat([image_tensor, padding], dim=2)
            img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = self.vae.clip.encode_video(image_tensor)

        return {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype),
        }

    def _generate_sync(
        self,
        session: SessionState,
        frame_callback: Optional[Callable[[int, int, str, int, int, str], None]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Dict[str, Any]:
        frames_requested = max(1, session.action_buffer.frames_recorded)
        keyboard_cond, mouse_cond, keyboard_cpu, mouse_cpu = session.action_buffer.export(self.device, self.weight_dtype)
        conditional_dict = {
            "cond_concat": session.cond_concat,
            "visual_context": session.visual_context,
            "keyboard_cond": keyboard_cond,
        }
        if session.action_space.enable_mouse and mouse_cond is not None:
            conditional_dict["mouse_cond"] = mouse_cond

        sampled_noise = torch.randn(
            [1, 16, self.args.max_num_output_frames, 44, 80],
            device=self.device,
            dtype=self.weight_dtype,
        )

        with torch.no_grad():
            def _action_provider(_start_frame: int) -> Optional[Dict[str, torch.Tensor]]:
                return None

            frames = self.pipeline.generate_next_frames(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                num_frames=frames_requested,
                initial_latent=None,
                mode=session.mode,
                action_provider=_action_provider,
            )

        total_frames = len(frames)
        if total_frames == 0:
            raise RuntimeError("Pipeline did not return any frames.")

        video_np = np.stack(frames, axis=0)
        video_np = np.ascontiguousarray(video_np)
        height, width = int(video_np.shape[1]), int(video_np.shape[2])

        if frame_callback is not None and loop is not None:
            for idx, frame in enumerate(frames):
                encoded_frame = self._encode_frame(frame)
                loop.call_soon_threadsafe(
                    frame_callback,
                    idx,
                    total_frames,
                    encoded_frame,
                    width,
                    height,
                    "jpeg",
                )

        keyboard_cfg = keyboard_cpu[:total_frames].float().numpy()
        config = (keyboard_cfg,)
        if session.action_space.enable_mouse and mouse_cpu is not None:
            mouse_cfg = mouse_cpu[:total_frames].float().numpy()
            config = (keyboard_cfg, mouse_cfg)

        base = session.output_stem
        base.parent.mkdir(parents=True, exist_ok=True)
        video_path = str(base.with_suffix(".mp4"))
        overlay_path = str(base.with_name(f"{base.name}_overlay").with_suffix(".mp4"))

        process_video(
            video_np,
            video_path,
            config,
            str(self.mouse_icon),
            mouse_scale=0.1,
            process_icon=False,
            mode=session.mode,
        )
        process_video(
            video_np,
            overlay_path,
            config,
            str(self.mouse_icon),
            mouse_scale=0.1,
            process_icon=True,
            mode=session.mode,
        )

        session.reset_actions()
        return {
            "video_path": video_path,
            "overlay_path": overlay_path,
            "frames_recorded": int(keyboard_cfg.shape[0]),
            "total_frames": total_frames,
            "frame_width": width,
            "frame_height": height,
        }

    def _encode_frame(self, frame: np.ndarray, format_: str = "jpeg") -> str:
        image = Image.fromarray(frame)
        buffer = BytesIO()
        if format_.lower() == "jpeg":
            image.save(buffer, format="JPEG", quality=85)
        else:
            image.save(buffer, format=format_.upper())
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _resize_crop(self, image, target_h: int, target_w: int):
        width, height = image.size
        if height / width > target_h / target_w:
            new_width = int(width)
            new_height = int(new_width * target_h / target_w)
        else:
            new_height = int(height)
            new_width = int(new_height * target_w / target_h)
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        return image.crop((left, top, right, bottom))


class WebsocketInferenceServer:
    """Bridges websocket events and the inference engine."""

    def __init__(self, inference: WebsocketInteractiveInference, host: str, port: int):
        self.inference = inference
        self.host = host
        self.port = port
        self.logger = logging.getLogger("matrix_game.websocket")
        self._server_root = Path(__file__).resolve().parent
        self._sessions: Dict[WebSocketServerProtocol, SessionState] = {}
        self.demo_images = self._discover_demo_images()

    async def run(self) -> None:
        self.logger.info("Starting websocket server on %s:%s", self.host, self.port)
        async with websockets.serve(self._handle_client, self.host, self.port, max_size=2 ** 22):
            await asyncio.Future()

    def _discover_demo_images(self) -> Dict[str, List[str]]:
        root = self._server_root / "demo_images"
        mapping: Dict[str, List[str]] = {}
        for mode in self.inference.action_spaces:
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

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        session: Optional[SessionState] = None
        await self._send(
            websocket,
            {
                "type": "ready",
                "modes": list(self.inference.action_spaces.keys()),
                "max_frames": self.inference.num_frames,
                "max_output_latents": self.inference.args.max_num_output_frames,
                "demo_images": self.demo_images,
            },
        )
        try:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "invalid_json", "Failed to decode JSON payload.")
                    continue

                msg_type = message.get("type")
                if msg_type == "setup":
                    try:
                        session = await self._handle_setup(websocket, message)
                    except Exception as exc:  # pylint: disable=broad-except
                        await self._send_error(websocket, "setup_failed", str(exc))
                elif msg_type == "action":
                    if not session:
                        await self._send_error(websocket, "no_session", "Call setup before sending actions.")
                        continue
                    frames = session.action_buffer.record_event(message)
                    await self._send(
                        websocket,
                        {
                            "type": "action_ack",
                            "session_id": session.session_id,
                            "frames_recorded": frames,
                            "frames_remaining": session.num_frames - frames,
                            "buffer_full": session.action_buffer.is_full(),
                        },
                    )
                elif msg_type == "generate":
                    if not session:
                        await self._send_error(websocket, "no_session", "Call setup before requesting generation.")
                        continue
                    def frame_callback(
                        index: int,
                        total: int,
                        encoded: str,
                        width: int,
                        height: int,
                        fmt: str,
                    ) -> None:
                        if websocket.closed:
                            return

                        async def _send_frame() -> None:
                            try:
                                await self._send(
                                    websocket,
                                    {
                                        "type": "frame",
                                        "session_id": session.session_id,
                                        "index": index,
                                        "total": total,
                                        "image": encoded,
                                        "format": fmt,
                                        "width": width,
                                        "height": height,
                                    },
                                )
                            except Exception as exc:  # pylint: disable=broad-except
                                self.logger.debug("Failed to stream frame %s: %s", index, exc)

                        asyncio.create_task(_send_frame())

                    try:
                        result = await self.inference.generate(session, frame_callback=frame_callback)
                    except Exception as exc:  # pylint: disable=broad-except
                        await self._send_error(websocket, "generation_failed", str(exc))
                        continue
                    await self._send(
                        websocket,
                        {
                            "type": "generation_complete",
                            "session_id": session.session_id,
                            **result,
                        },
                    )
                elif msg_type == "reset":
                    if not session:
                        await self._send_error(websocket, "no_session", "Nothing to reset.")
                        continue
                    image_override = message.get("image_path")
                    mode_override = message.get("mode")
                    if image_override or mode_override:
                        try:
                            session = await self._handle_setup(
                                websocket,
                                {
                                    "type": "setup",
                                    "image_path": image_override or session.image_path,
                                    "mode": mode_override or session.mode,
                                    "session_id": session.session_id,
                                },
                            )
                        except Exception as exc:  # pylint: disable=broad-except
                            await self._send_error(websocket, "reset_failed", str(exc))
                    else:
                        session.reset_actions()
                        await self._send(
                            websocket,
                            {
                                "type": "session_reset",
                                "session_id": session.session_id,
                                "frames_recorded": 0,
                                "frames_remaining": session.num_frames,
                            },
                        )
                elif msg_type == "ping":
                    await self._send(websocket, {"type": "pong"})
                else:
                    await self._send_error(websocket, "unsupported", f"Unknown message type '{msg_type}'.")
        except ConnectionClosed:
            self.logger.info("Client disconnected")
        finally:
            if session:
                self._sessions.pop(websocket, None)

    async def _handle_setup(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]) -> SessionState:
        image_path = message.get("image_path")
        if not image_path:
            raise ValueError("`image_path` is required in setup messages.")
        mode = message.get("mode")
        session_id = message.get("session_id")

        loop = asyncio.get_running_loop()
        session = await loop.run_in_executor(None, self.inference.prepare_session, image_path, mode, session_id)
        self._sessions[websocket] = session
        await self._send(
            websocket,
            {
                "type": "session_started",
                "session_id": session.session_id,
                "mode": session.mode,
                "num_frames": session.num_frames,
            },
        )
        return session

    async def _send(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]) -> None:
        await websocket.send(json.dumps(payload))

    async def _send_error(self, websocket: WebSocketServerProtocol, code: str, message: str) -> None:
        await self._send(websocket, {"type": "error", "error": {"code": code, "message": message}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml", help="Path to OmegaConf config file.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Optional diffusion checkpoint in safetensors format.")
    parser.add_argument("--output_folder", type=str, default="outputs/", help="Output directory for rendered videos.")
    parser.add_argument("--max_num_output_frames", type=int, default=360, help="Max number of latent frames (matches original CLI).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0", help="Directory containing Wan VAE weights.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Websocket bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Websocket bind port.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    set_seed(args.seed)
    inference = WebsocketInteractiveInference(args)
    server = WebsocketInferenceServer(inference, args.host, args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
