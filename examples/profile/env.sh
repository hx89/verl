#!/usr/bin/env bash
# Per-user overrides for config_b2_interactive.sh. Sourced by the interactive
# script at startup (see the `[ -f "${SCRIPT_DIR}/env.sh" ] && source ...`
# hook near the top). Anything exported here is picked up by the script's
# `${VAR:-default}` fallbacks for VERL_SRC_DIR, HF_HOME, and OUT_DIR.
#
# This file is NOT meant to be committed to shared branches — it captures the
# local container layout, which differs per developer.
#
# Container mount assumption: the host directory
#   /home/scratch.mdfahimfaysa_gpu
# is bind-mounted into the container at /opt/workspace, so everything under
# /home/scratch.mdfahimfaysa_gpu/projects/verl-exp/ on the host appears at
# /opt/workspace/projects/verl-exp/ inside the container.

# Anchor everything to one folder so verl source, HF cache, and run outputs
# all live together and survive container relaunches.
VERL_EXP_ROOT="/opt/workspace/projects/verl-exp"

# -------- VERL source location --------
# The host repo at $VERL_EXP_ROOT/verl is mounted through to the same path
# inside the container.
export VERL_SRC_DIR="${VERL_EXP_ROOT}/verl"

# -------- HuggingFace cache --------
# Stores tokenizer + Qwen3-30B-A3B-Base BF16 checkpoint (~60 GB). Make sure
# the underlying volume has room before launching.
export HF_HOME="${VERL_EXP_ROOT}/hf_cache"

# -------- Output directory (logs + nsys profiles) --------
export OUT_DIR="${VERL_EXP_ROOT}/logs"

# -------- Container image tag (logged to wandb job_info) --------
# Purely metadata — gets recorded in the wandb run config as
# job_info.verl_container_image. Set this to whatever image you actually
# launched so the wandb log is accurate.
export VERL_CONTAINER_IMAGE="verlai/verl:vllm017.latest"

# -------- Secrets --------
# HF_TOKEN and WANDB_API_KEY are intentionally NOT stored here. Pass them in
# the environment at invocation instead (a leading space keeps them out of
# shell history):
#    HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_b2_interactive.sh <exp> <mode>