import asyncio
import base64
import contextlib
import json
import socket
import sys
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import util
from io import BytesIO
from pathlib import Path

import pytest
import torch
import websockets
from PIL import Image


if "diffusers" not in sys.modules:
    diffusers_module = types.ModuleType("diffusers")
    diffusers_utils = types.ModuleType("diffusers.utils")

    def _load_image_stub(path):
        if Path(path).exists():
            return Image.open(path)
        return Image.new("RGB", (8, 8))

    diffusers_utils.load_image = _load_image_stub
    diffusers_module.utils = diffusers_utils
    config_utils = types.ModuleType("diffusers.configuration_utils")
    loaders_module = types.ModuleType("diffusers.loaders")
    models_module = types.ModuleType("diffusers.models")
    modeling_utils = types.ModuleType("diffusers.models.modeling_utils")

    class _ConfigMixin:
        pass

    def _register_to_config(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    config_utils.ConfigMixin = _ConfigMixin
    config_utils.register_to_config = _register_to_config

    class _FromOriginalModelMixin:
        pass

    class _PeftAdapterMixin:
        pass

    loaders_module.FromOriginalModelMixin = _FromOriginalModelMixin
    loaders_module.PeftAdapterMixin = _PeftAdapterMixin

    class _ModelMixin:
        pass

    modeling_utils.ModelMixin = _ModelMixin
    models_module.modeling_utils = modeling_utils

    sys.modules["diffusers"] = diffusers_module
    sys.modules["diffusers.utils"] = diffusers_utils
    sys.modules["diffusers.configuration_utils"] = config_utils
    sys.modules["diffusers.loaders"] = loaders_module
    sys.modules["diffusers.models"] = models_module
    sys.modules["diffusers.models.modeling_utils"] = modeling_utils
    diffusers_module.configuration_utils = config_utils
    diffusers_module.loaders = loaders_module
    diffusers_module.models = models_module

if "einops" not in sys.modules:
    einops_module = types.ModuleType("einops")
    einops_module.rearrange = lambda tensor, *_, **__: tensor
    einops_module.repeat = lambda tensor, *_, **__: tensor
    sys.modules["einops"] = einops_module

if "omegaconf" not in sys.modules:
    omegaconf_module = types.ModuleType("omegaconf")

    class _OmegaConf:
        @staticmethod
        def load(path):
            return {}

    omegaconf_module.OmegaConf = _OmegaConf
    sys.modules["omegaconf"] = omegaconf_module

if "safetensors" not in sys.modules:
    safetensors_module = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda *_args, **_kwargs: {}
    safetensors_module.torch = safetensors_torch
    sys.modules["safetensors"] = safetensors_module
    sys.modules["safetensors.torch"] = safetensors_torch

if "torchvision.transforms.v2" not in sys.modules:
    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")

    class _IdentityTransform:
        def __call__(self, value):
            return value

    class _Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    v2_module = types.SimpleNamespace(
        Compose=lambda transforms: _Compose(transforms),
        Resize=lambda *_, **__: _IdentityTransform(),
        ToTensor=lambda *_, **__: _IdentityTransform(),
        Normalize=lambda *_, **__: _IdentityTransform(),
    )

    torchvision_module.transforms = transforms_module
    transforms_module.v2 = v2_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module
    sys.modules["torchvision.transforms.v2"] = v2_module

if "easydict" not in sys.modules:
    easydict_module = types.ModuleType("easydict")

    class EasyDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as err:
                raise AttributeError(name) from err

        def __setattr__(self, name, value):
            self[name] = value

    easydict_module.EasyDict = EasyDict
    sys.modules["easydict"] = easydict_module

if "flash_attn" not in sys.modules:
    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.flash_attn_func = lambda *args, **kwargs: None
    sys.modules["flash_attn"] = flash_attn_module

if "ftfy" not in sys.modules:
    ftfy_module = types.ModuleType("ftfy")
    ftfy_module.fix_text = lambda text: text
    sys.modules["ftfy"] = ftfy_module

if "regex" not in sys.modules:
    import re as _re

    sys.modules["regex"] = _re

if "pipeline" not in sys.modules:
    pipeline_module = types.ModuleType("pipeline")

    class _StubPipeline:
        def __init__(self, *args, **kwargs):
            self.generator = types.SimpleNamespace(load_state_dict=lambda *_a, **_k: None)

        def to(self, *args, **kwargs):
            return self

        def generate_next_frames(self, *args, **kwargs):
            return []

    pipeline_module.CausalInferenceStreamingPipeline = _StubPipeline
    sys.modules["pipeline"] = pipeline_module

if "utils.wan_wrapper" not in sys.modules:
    utils_module = types.ModuleType("utils")
    wan_wrapper_module = types.ModuleType("utils.wan_wrapper")

    class _WanDiffusionWrapper:
        def __init__(self, *args, **kwargs):
            pass

    wan_wrapper_module.WanDiffusionWrapper = _WanDiffusionWrapper
    utils_module.wan_wrapper = wan_wrapper_module
    sys.modules["utils"] = utils_module
    sys.modules["utils.wan_wrapper"] = wan_wrapper_module

if "utils.misc" not in sys.modules:
    misc_module = types.ModuleType("utils.misc")
    misc_module.set_seed = lambda *_args, **_kwargs: None
    sys.modules["utils.misc"] = misc_module

if "utils.visualize" not in sys.modules:
    visualize_module = types.ModuleType("utils.visualize")
    visualize_module.process_video = lambda *args, **kwargs: None
    sys.modules["utils.visualize"] = visualize_module

if "demo_utils.vae_block3" not in sys.modules:
    demo_utils_module = types.ModuleType("demo_utils")
    vae_block_module = types.ModuleType("demo_utils.vae_block3")

    class _VAEDecoderWrapper:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def requires_grad_(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        def compile(self, *args, **kwargs):
            return None

        def load_state_dict(self, *args, **kwargs):
            return None

    vae_block_module.VAEDecoderWrapper = _VAEDecoderWrapper
    demo_utils_module.vae_block3 = vae_block_module
    sys.modules["demo_utils"] = demo_utils_module
    sys.modules["demo_utils.vae_block3"] = vae_block_module

if "wan.vae.wanx_vae" not in sys.modules:
    wan_module = types.ModuleType("wan")
    wan_vae_module = types.ModuleType("wan.vae")
    wanx_vae_module = types.ModuleType("wan.vae.wanx_vae")

    class _VAEWrapper:
        def to(self, *args, **kwargs):
            return self

        def requires_grad_(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

    class _ClipStub:
        def encode_video(self, *args, **kwargs):
            return torch.zeros(1)

    def _get_wanx_vae_wrapper(*_args, **_kwargs):
        wrapper = _VAEWrapper()
        wrapper.clip = _ClipStub()
        wrapper.encode = lambda *a, **k: torch.zeros(1)
        return wrapper

    wanx_vae_module.get_wanx_vae_wrapper = _get_wanx_vae_wrapper
    wan_vae_module.wanx_vae = wanx_vae_module
    wan_module.vae = wan_vae_module
    sys.modules["wan"] = wan_module
    sys.modules["wan.vae"] = wan_vae_module
    sys.modules["wan.vae.wanx_vae"] = wanx_vae_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULE_PATH = Path(__file__).resolve().parents[2] / "websocket_inference_server.py"
SPEC = util.spec_from_file_location("matrix_game_websocket_server", MODULE_PATH)
websocket_module = util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(websocket_module)  # type: ignore[assignment]

ActionBuffer = websocket_module.ActionBuffer
ActionSpace = websocket_module.ActionSpace
SessionState = websocket_module.SessionState
WebsocketInferenceServer = websocket_module.WebsocketInferenceServer


@dataclass
class FakeArgs:
    max_num_output_frames: int = 4
    output_folder: str = "/tmp/matrix_game_test"


class FakeInference:
    """Minimal inference stub that mimics the server contract without heavy deps."""

    def __init__(self, output_root: Path):
        self.args = FakeArgs(output_folder=str(output_root))
        self.device = torch.device("cpu")
        self.num_frames = 4
        self._output_root = output_root
        self.action_spaces = {
            "test": ActionSpace(
                mode="test",
                keyboard_map={"forward": 0},
                default_keys=("forward",),
                enable_mouse=False,
            )
        }

    def prepare_session(self, image_path, mode, session_id=None):
        space = self.action_spaces["test"]
        buffer = ActionBuffer(space, self.num_frames)
        output_stem = (self._output_root / (session_id or "stub1234")).with_suffix("")
        return SessionState(
            session_id=session_id or "stub1234",
            mode=space.mode,
            image_path=image_path,
            cond_concat=torch.zeros(1),
            visual_context=torch.zeros(1),
            action_space=space,
            action_buffer=buffer,
            num_frames=self.num_frames,
            output_stem=output_stem,
        )

    async def generate(self, session, frame_callback=None):
        total = max(1, session.action_buffer.frames_recorded)
        image = Image.new("RGB", (8, 8), color=(0, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        if frame_callback is not None:
            for index in range(total):
                frame_callback(index, total, encoded, 8, 8, "jpeg")
        session.reset_actions()
        return {
            "video_path": str(self._output_root / "stub.mp4"),
            "overlay_path": str(self._output_root / "stub_overlay.mp4"),
            "frames_recorded": total,
            "total_frames": total,
            "frame_width": 8,
            "frame_height": 8,
        }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def running_server(inference, host: str, port: int):
    server = WebsocketInferenceServer(inference, host, port)
    task = asyncio.create_task(server.run())
    try:
        yield f"ws://{host}:{port}"
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_websocket_integration_round_trip(tmp_path):
    port = _free_port()
    inference = FakeInference(tmp_path)
    async with running_server(inference, "127.0.0.1", port) as uri:
        connection = None
        for _ in range(40):
            try:
                connection = await websockets.connect(uri, ping_interval=None)
                break
            except OSError:
                await asyncio.sleep(0.05)
        if connection is None:
            pytest.fail("Failed to connect to websocket server")

        async with connection as ws:
            ready = json.loads(await ws.recv())
            assert ready["type"] == "ready"
            assert "test" in ready["modes"]

            await ws.send(
                json.dumps(
                    {"type": "setup", "image_path": "demo.png", "mode": "test", "session_id": "case1"}
                )
            )
            started = json.loads(await ws.recv())
            assert started["type"] == "session_started"
            assert started["session_id"] == "case1"

            await ws.send(json.dumps({"type": "action", "keyboard": {"state": ["forward"]}}))
            ack = json.loads(await ws.recv())
            assert ack["type"] == "action_ack"
            assert ack["frames_recorded"] == 1

            await ws.send(json.dumps({"type": "generate"}))
            seen_frame = False
            while True:
                message = json.loads(await ws.recv())
                if message["type"] == "frame":
                    seen_frame = True
                    assert message["session_id"] == "case1"
                    assert message["total"] >= 1
                elif message["type"] == "generation_complete":
                    assert message["session_id"] == "case1"
                    break
            assert seen_frame
