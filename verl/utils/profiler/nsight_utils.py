# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for Ray + Nsight Systems profiling in VERL."""

from __future__ import annotations

import glob
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# Megatron hybrid worker discrete nsys ranges in VERL:
# ref_compute_log_prob, actor_compute_log_prob, actor_update.
DEFAULT_DISCRETE_SUBTASKS_PER_STEP = 3


def configure_nsight_options(
    save_path: Optional[str],
    worker_nsight_options: Optional[dict],
    profile_steps: Optional[list[int]],
    discrete: bool = False,
    discrete_subtasks_per_step: int = DEFAULT_DISCRETE_SUBTASKS_PER_STEP,
) -> dict:
    """Return worker nsight options with output path and capture-range-end configured.

    Ray prepends ``/tmp/ray/session_*/logs/nsight/`` to relative ``-o`` paths. Use an
    absolute path so reports are written directly under the experiment directory.
    """
    options = dict(worker_nsight_options or {})

    if save_path:
        # Single profile step: write directly under step_<N>/ to avoid a duplicate
        # flat copy in nsys_profiles/ plus a collect copy in step_<N>/.
        if profile_steps and len(profile_steps) == 1:
            step_dir = os.path.join(save_path, "nsys_profiles", f"step_{profile_steps[0]}")
        else:
            step_dir = os.path.join(save_path, "nsys_profiles")
        os.makedirs(step_dir, exist_ok=True)
        # nsys appends .nsys-rep; %p expands to PID, RID suffix added per capture range.
        options.setdefault("o", os.path.join(step_dir, "worker_process_%p"))

    if options.get("capture-range-end") is None and profile_steps:
        num_captures = len(profile_steps)
        if discrete:
            num_captures *= discrete_subtasks_per_step
        else:
            num_captures *= 1
        options["capture-range-end"] = f"repeat-shutdown:{num_captures}"

    return options


def configure_controller_nsight_options(
    save_path: Optional[str],
    controller_nsight_options: Optional[dict],
    profile_steps: Optional[list[int]] = None,
) -> dict:
    """Return controller nsight options with an absolute output path when possible."""
    options = dict(controller_nsight_options or {})
    if save_path:
        if profile_steps and len(profile_steps) == 1:
            step_dir = os.path.join(save_path, "nsys_profiles", f"step_{profile_steps[0]}")
        else:
            step_dir = os.path.join(save_path, "nsys_profiles")
        os.makedirs(step_dir, exist_ok=True)
        options.setdefault("o", os.path.join(step_dir, "controller_process_%p"))
    return options


def collect_nsys_profiles(save_path: str, step: int) -> int:
    """Move any ``*.nsys-rep`` from flat ``nsys_profiles/`` or Ray defaults into ``step_<step>/``.

    When ``configure_nsight_options`` targets a single profile step, reports are already
    written under ``step_<step>/`` and this mainly handles ``/tmp/ray/.../nsight`` fallback.
    For multi-step runs, flat ``nsys_profiles/*.nsys-rep`` files are moved (not copied)
    into the per-step directory after each profiled step. Returns the number of files moved.
    """
    dest_dir = os.path.join(save_path, "nsys_profiles", f"step_{step}")
    os.makedirs(dest_dir, exist_ok=True)

    moved = 0
    seen: set[str] = set()

    # Flat nsys_profiles/ root (multi-step ``-o`` or legacy layout).
    for src in glob.glob(os.path.join(save_path, "nsys_profiles", "*.nsys-rep")):
        moved += _move_if_new(src, dest_dir, seen)

    # Ray default location(s).
    for nsight_dir in sorted(glob.glob("/tmp/ray/session_*/logs/nsight")):
        for src in glob.glob(os.path.join(nsight_dir, "*.nsys-rep")):
            moved += _move_if_new(src, dest_dir, seen)

    if moved:
        logger.info("Collected %d nsys profile(s) into %s", moved, dest_dir)
    elif not glob.glob(os.path.join(dest_dir, "*.nsys-rep")):
        logger.warning(
            "No nsys profiles found to collect for step %s (checked %s/nsys_profiles and /tmp/ray/session_*/logs/nsight)",
            step,
            save_path,
        )
    return moved


def _move_if_new(src: str, dest_dir: str, seen: set[str]) -> int:
    real_src = os.path.realpath(src)
    if real_src in seen:
        return 0
    seen.add(real_src)
    dest = os.path.join(dest_dir, os.path.basename(src))
    if os.path.realpath(dest) == real_src:
        return 0
    if os.path.exists(dest):
        os.remove(src)
        return 0
    shutil.move(src, dest)
    return 1
