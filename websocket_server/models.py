"""Shared models and buffers for Matrix-Game FastAPI inference service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class ActionSpace:
    """Mapping from symbolic keyboard/mouse inputs to dense tensors."""

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
        self.keyboard = torch.zeros(
            (num_frames, space.keyboard_dim), dtype=torch.float32
        )
        self.mouse = (
            torch.zeros((num_frames, 2), dtype=torch.float32)
            if space.enable_mouse
            else None
        )
        self._cursor = 0
        self._last_keys: Tuple[str, ...] = space.default_keys
        self._last_mouse_vec = (
            torch.zeros(2, dtype=torch.float32) if space.enable_mouse else None
        )

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
                mouse_vec = (
                    self._last_mouse_vec
                    if self._last_mouse_vec is not None
                    else torch.zeros(2, dtype=torch.float32)
                )
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
        self._last_mouse_vec = (
            torch.zeros(2, dtype=torch.float32) if self.space.enable_mouse else None
        )

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
                fill = (
                    self._last_mouse_vec
                    if self._last_mouse_vec is not None
                    else torch.zeros(2, dtype=torch.float32)
                )
                for idx in range(self.num_frames):
                    self.mouse[idx] = fill
            self._cursor = self.num_frames
            return

        last_key_row = self.keyboard[self._cursor - 1].clone()
        last_mouse_row = (
            self.mouse[self._cursor - 1].clone()
            if self.space.enable_mouse and self.mouse is not None
            else None
        )
        for idx in range(self._cursor, self.num_frames):
            self.keyboard[idx] = last_key_row
            if last_mouse_row is not None:
                self.mouse[idx] = last_mouse_row
        self._cursor = self.num_frames


@dataclass
class CachedConditions:
    """CPU resident cache entries shared across GPU workers."""

    cond_concat: torch.Tensor
    visual_context: torch.Tensor

    def to_device(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cond_concat.to(device=device, dtype=dtype, non_blocking=True),
            self.visual_context.to(device=device, dtype=dtype, non_blocking=True),
        )


@dataclass
class SessionState:
    """Runtime metadata for a single client session."""

    session_id: str
    mode: str
    image_path: str
    action_space: ActionSpace
    action_buffer: ActionBuffer
    num_frames: int
    output_stem: Path
    conditions_key: str

    @classmethod
    def new(
        cls,
        image_path: str,
        mode: str,
        action_space: ActionSpace,
        num_frames: int,
        output_dir: Path,
        conditions_key: str,
        session_id: Optional[str] = None,
    ) -> "SessionState":
        session_uuid = session_id or uuid.uuid4().hex[:8]
        buffer = ActionBuffer(action_space, num_frames)
        output_stem = (output_dir / session_uuid).with_suffix("")
        return cls(
            session_id=session_uuid,
            mode=mode,
            image_path=image_path,
            action_space=action_space,
            action_buffer=buffer,
            num_frames=num_frames,
            output_stem=output_stem,
            conditions_key=conditions_key,
        )

    def reset_actions(self) -> None:
        self.action_buffer.reset()
