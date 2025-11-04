"""GPU worker that performs World Model inference."""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import v2

from demo_utils.vae_block3 import VAEDecoderWrapper
from imageops import load_image
from pipeline import CausalInferenceStreamingPipeline
from utils.visualize import process_video
from vae.model import WanDiffusionWrapper

from .models import ActionSpace, CachedConditions, SessionState
from modules import clip


@abs.abstract
class VAEWrapper:
    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.vae, name)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, latents):
        return NotImplementedError


class WanxVAEWrapper(VAEWrapper):
    def __init__(self, vae, clip):
        super(WanxVAEWrapper, self).__init__()
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.clip = clip
        if clip is not None:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def encode(self, x, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        x = self.vae.encode(
            x, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )  # already scaled
        return x  # torch.stack(x, dim=0)

    def clip_img(self, x):
        x = self.clip(x)
        return x

    def decode(
        self, latents, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        videos = self.vae.decode(
            latents,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return videos  # self.vae.decode(videos, dim=0) # already scaled

    def to(self, device, dtype):
        # 移动 vae 到指定设备
        self.vae = self.vae.to(device, dtype)

        # 如果 clip 存在，也移动到指定设备
        if self.clip is not None:
            self.clip = self.clip.to(device, dtype)

        return self


def get_wanx_vae_wrapper(model_path, weight_dtype):
    vae = WanVAE(pretrained_path=os.path.join(model_path, "Wan2.1_VAE.pth")).to(
        weight_dtype
    )

    clip_model = clip.CLIPModel(
        checkpoint_path=os.path.join(
            model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        ),
        tokenizer_path=os.path.join(model_path, "xlm-roberta-large"),
    )
    return WanxVAEWrapper(vae, clip_model)


@dataclass(frozen=True)
class WorkerOptions:
    """Configuration shared by inference workers."""

    config_path: str
    checkpoint_path: str
    output_folder: str
    max_num_output_frames: int
    pretrained_model_path: str


class MatrixGameInferenceWorker:
    """Owns model weights and performs GPU-accelerated generation."""

    def __init__(self, options: WorkerOptions, device: str):
        self.options = options
        self.device = torch.device(device)
        self.weight_dtype = torch.bfloat16
        self.output_dir = Path(self.options.output_folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.num_frames = (self.options.max_num_output_frames - 1) * 4 + 1
        self.mouse_icon = (
            Path(__file__).resolve().parent.parent / "assets" / "images" / "mouse.png"
        )
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
        self.config = OmegaConf.load(self.options.config_path)
        self.default_mode = (
            self.config.pop("mode") if "mode" in self.config else "universal"
        )

    def _init_models(self) -> None:
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True
        )
        vae_decoder = VAEDecoderWrapper()
        vae_state = torch.load(
            Path(self.options.pretrained_model_path) / "Wan2.1_VAE.pth",
            map_location="cpu",
        )
        decoder_state = {
            key: value
            for key, value in vae_state.items()
            if "decoder." in key or "conv2" in key
        }
        vae_decoder.load_state_dict(decoder_state)
        vae_decoder.to(self.device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        pipeline = CausalInferenceStreamingPipeline(
            self.config, generator=generator, vae_decoder=vae_decoder
        )
        if self.options.checkpoint_path:
            state_dict = load_file(self.options.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.options.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    async def encode_conditionals(self, image_path: str) -> CachedConditions:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._encode_conditionals_sync, image_path
        )

    def _encode_conditionals_sync(self, image_path: str) -> CachedConditions:
        tiler_kwargs = {
            "tiled": True,
            "tile_size": [44, 80],
            "tile_stride": [23, 38],
        }

        image = load_image(image_path)
        image = self._resize_crop(image, 352, 640)
        with torch.no_grad():
            image_tensor = self.frame_process(image)[None, :, None, :, :].to(
                dtype=self.weight_dtype, device=self.device
            )
            padding = torch.zeros_like(image_tensor).repeat(
                1, 1, 4 * (self.options.max_num_output_frames - 1), 1, 1
            )
            img_cond = torch.cat([image_tensor, padding], dim=2)
            img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(
                self.device
            )
            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = self.vae.clip.encode_video(image_tensor)

        return CachedConditions(
            cond_concat=cond_concat.to(device="cpu"),
            visual_context=visual_context.to(device="cpu"),
        )

    async def generate(
        self,
        session: SessionState,
        conditions: CachedConditions,
        frame_callback: Optional[Callable[[int, int, str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        async with self.generation_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._generate_sync, session, conditions, frame_callback, loop
            )

    def _generate_sync(
        self,
        session: SessionState,
        conditions: CachedConditions,
        frame_callback: Optional[Callable[[int, int, str, int, int, str], None]],
        loop: asyncio.AbstractEventLoop,
    ) -> Dict[str, Any]:
        frames_requested = max(1, session.action_buffer.frames_recorded)
        keyboard_cond, mouse_cond, keyboard_cpu, mouse_cpu = (
            session.action_buffer.export(self.device, self.weight_dtype)
        )

        cond_concat, visual_context = conditions.to_device(
            self.device, self.weight_dtype
        )
        conditional_dict = {
            "cond_concat": cond_concat,
            "visual_context": visual_context,
            "keyboard_cond": keyboard_cond,
        }
        if session.action_space.enable_mouse and mouse_cond is not None:
            conditional_dict["mouse_cond"] = mouse_cond

        sampled_noise = torch.randn(
            [1, 16, self.options.max_num_output_frames, 44, 80],
            device=self.device,
            dtype=self.weight_dtype,
        )

        with torch.no_grad():

            def _action_provider(
                _start_frame: int,
            ) -> Optional[Dict[str, torch.Tensor]]:
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

        if frame_callback is not None:
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
