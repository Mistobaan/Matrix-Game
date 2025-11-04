"""High-level coordination for Matrix-Game inference workers."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .cache import InferenceCache
from .models import ActionSpace, SessionState
from .worker import MatrixGameInferenceWorker, WorkerOptions


@dataclass(frozen=True)
class ServiceOptions:
    """Configuration for building the FastAPI inference service."""

    config_path: str
    checkpoint_path: str
    output_folder: str
    max_num_output_frames: int
    pretrained_model_path: str
    devices: Iterable[str]


class MatrixGameInferenceService:
    """Coordinates GPU workers and shared caches."""

    def __init__(self, options: ServiceOptions):
        devices = list(options.devices)
        if not devices:
            raise ValueError("At least one device must be specified.")

        worker_opts = WorkerOptions(
            config_path=options.config_path,
            checkpoint_path=options.checkpoint_path,
            output_folder=options.output_folder,
            max_num_output_frames=options.max_num_output_frames,
            pretrained_model_path=options.pretrained_model_path,
        )
        self.workers: List[MatrixGameInferenceWorker] = [
            MatrixGameInferenceWorker(worker_opts, device=device) for device in devices
        ]
        self._worker_cycle = itertools.cycle(self.workers)
        self._cycle_lock = asyncio.Lock()

        anchor_worker = self.workers[0]
        self.default_mode = anchor_worker.default_mode
        self.action_spaces: Dict[str, ActionSpace] = anchor_worker.action_spaces
        self.num_frames = anchor_worker.num_frames
        self.output_dir = Path(options.output_folder)

        self.cache = InferenceCache()

    async def prepare_session(
        self,
        image_path: str,
        mode: Optional[str],
        session_id: Optional[str] = None,
    ) -> SessionState:
        target_mode = mode or self.default_mode
        if target_mode not in self.action_spaces:
            raise ValueError(
                f"Unsupported mode '{target_mode}'. Valid modes: {list(self.action_spaces)}"
            )

        action_space = self.action_spaces[target_mode]
        conditions_key = InferenceCache.make_key(image_path, target_mode)

        async def _factory():
            return await self.workers[0].encode_conditionals(image_path)

        await self.cache.get_or_create(conditions_key, _factory)

        return SessionState.new(
            image_path=image_path,
            mode=target_mode,
            action_space=action_space,
            num_frames=self.num_frames,
            output_dir=self.output_dir,
            conditions_key=conditions_key,
            session_id=session_id,
        )

    async def generate(
        self,
        session: SessionState,
        frame_callback: Optional[Callable[[int, int, str, int, int, str], None]] = None,
    ):
        conditions = await self.cache.get(session.conditions_key)
        if conditions is None:

            async def _factory():
                return await self.workers[0].encode_conditionals(session.image_path)

            conditions = await self.cache.get_or_create(
                session.conditions_key, _factory
            )

        worker = await self._next_worker()
        return await worker.generate(session, conditions, frame_callback)

    async def _next_worker(self) -> MatrixGameInferenceWorker:
        async with self._cycle_lock:
            return next(self._worker_cycle)
