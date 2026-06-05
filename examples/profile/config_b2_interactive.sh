#!/usr/bin/env bash
# config_b2: Mid-batch-size GRPO on a single 8xH100 DGX node with
# Qwen3-30B-A3B-Base on gsm8k. The second (and last) profileable point in the
# host offloading study.
#
# Topology / hyperparameters are kept identical to config_a / config_b1 so
# that the only varied axis across (a, b1, b2) is the number of microbatches
# per train step. Specifically, this config runs:
#   - 8 prompts/batch, 2 prompts/minibatch, 16 responses/prompt
#   - microbatch size = 8 sequences (fixed, no dynamic batching)
#   - => 4 minibatches/step, 4 microbatches/minibatch, 16 microbatches/step
#
# HF_TOKEN and WANDB_API_KEY are read from the environment (not stored in
# env.sh). Pass them at invocation; a leading space keeps them out of history:
#    HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_b2_interactive.sh <exp> <mode>

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/env.sh" ] && source "${SCRIPT_DIR}/env.sh"

# Require the secrets to be supplied via the environment. Validate with xtrace
# disabled so the token values are never echoed into the tee'd log.
{ set +x; } 2>/dev/null
: "${HF_TOKEN:?not set — pass it at invocation, e.g. HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_b2_interactive.sh <exp> <mode>}"
: "${WANDB_API_KEY:?not set — pass it at invocation, e.g. HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_b2_interactive.sh <exp> <mode>}"
set -x

# Driver-side env vars inherited from the reference script. These are also
# passed to VERL worers via $RAY_KWARGS below.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/opt/workspace/projects/verl-exp/verl:${PYTHONPATH:-}

# HF cache on local node storage; avoids filling networked home directories.
# Override via env.sh by exporting HF_HOME before this script's section runs.
export HF_HOME="${HF_HOME:-/opt/hf-cache}"
mkdir -p $HF_HOME

# ===================================== Topology =====================================
NNODES=1
GPUS_PER_NODE=8

# ===================================== Output / metadata =====================================
# Experiment name: first positional arg (base name). Profiling mode: second arg
# nsys | torch | none (default: nsys). PROFILE_MODE is appended to the run dir
# unless the base name already ends with _nsys, _torch, or _none:
#   ./config_b2_interactive.sh vllm17_may31 nsys   -> logs/vllm17_may31_nsys/
#   ./config_b2_interactive.sh vllm17_may31 torch  -> logs/vllm17_may31_torch/
WANDB_PROJECT_NAME="verl_grpo_gsm8k"
EXPERIMENT_NAME_BASE="${1:-config_b2-qwen_3_30b_a3b_base-gsm8k-$(date +%Y_%m_%d)}"
PROFILE_MODE="${2:-${PROFILE_MODE:-nsys}}"
case "${EXPERIMENT_NAME_BASE}" in
  *_nsys|*_torch|*_none) EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}" ;;
  *) EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}_${PROFILE_MODE}" ;;
esac
OUT_DIR="${OUT_DIR:-${PWD}/logs}"
EXP_DIR="${OUT_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${EXP_DIR}"
OUT_FILE="${EXP_DIR}/${EXPERIMENT_NAME}.log"
total_training_steps=15

VERL_SRC_DIR="${VERL_SRC_DIR:-/opt/workspace/RL/verl}"
# Avoid "dubious ownership" errors that occur with mounted volumes in Docker.
git config --global --add safe.directory "${VERL_SRC_DIR}"
VERL_COMMIT=$(git -C "${VERL_SRC_DIR}" rev-parse HEAD)
echo "Using VERL Commit: ${VERL_COMMIT}"

# Adds info to the job config logged to WandB so that data analysis scripts
# work correctly.
JOB_INFO=(
    "+job_info.config_number='b2'"
    "+job_info.cluster='computelab'"
    "+job_info.gpu='B200-SXM'"
    "+job_info.cpu='2x 56-core x86 Intel_Xeon_Platinum_8570'"
    "+job_info.host_ram='2.0 TiB'"
    "+job_info.verl_container_image='${VERL_CONTAINER_IMAGE:-unknown}'"
    "+job_info.verl_commit=${VERL_COMMIT}"
    "+job_info.environment_name='gsm8k'"
    "+job_info.max_num_turns=1"
    "+job_info.environment_type='toy,math'"
    "+job_info.architecture_name='Qwen3-30B-A3B-Base'"
    "+job_info.num_params='30B'"
    "+job_info.algorithm_name='grpo'"
    "+job_info.colocated=true"
    "+job_info.async_lag=0"
    "+job_info.log_dir=${EXP_DIR}"
    "+job_info.profile_mode=${PROFILE_MODE}"
)

# ===================================== Dataset / model =====================================
cd ${VERL_SRC_DIR}
DATA_SAVE_DIR=~/data/gsm8k
if [[ ! -f ${DATA_SAVE_DIR}/train.parquet || ! -f ${DATA_SAVE_DIR}/test.parquet ]]; then
    python examples/data_preprocess/gsm8k.py --local_save_dir ${DATA_SAVE_DIR}
fi
TRAIN_FILE=${DATA_SAVE_DIR}/train.parquet
TEST_FILE=${DATA_SAVE_DIR}/test.parquet

HUGGINGFACE_MODEL_PATH="Qwen/Qwen3-30B-A3B-Base"
hf download "${HUGGINGFACE_MODEL_PATH}"

# ===================================== Algorithm =====================================
adv_estimator=grpo

# Reference-policy KL handling. KL-in-reward is disabled; KL-as-loss is
# enabled at a small coefficient — standard GRPO setup.
use_kl_in_reward=False
kl_coef=0.0                 # Used when use_kl_in_reward=True.
use_kl_loss=True
kl_loss_coef=0.001          # Used when use_kl_loss=True.

# Kept from reference script.
actor_lr=1e-6
actor_lr_warmup_steps=10
actor_weight_decay=0.1
actor_clip_grad=1.0
actor_entropy_coeff=0
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1          # 0 for HF rollout, -1 for vLLM rollout.
val_temperature=1.0
val_top_p=0.7
val_top_k=-1

# ===================================== Data =====================================
max_prompt_length=512
max_response_length=1024

train_prompt_bsz=8
train_prompt_mini_bsz=2
n_resp_per_prompt=16
n_resp_per_prompt_val=1

use_dynamic_bsz=False
ppo_micro_batch_size_per_gpu=8
infer_logprob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}


# ===================================== Megatron actor =====================================
# TP=8,EP=8,ETP=1,PP=1. Moves from the upstream 4-node TP=1,EP=8 layout to a
# single-node TP=8,EP=8 layout.
#
# A note about host offloading: The offloading flags DO NOT correspond to
# pipelined offloading during the train step, but to coarse-grained offloading
# where the specified components of the train state are offloaded to the host
# when inference workers start executing.
#
# A note about gradient checkpointing: This is configured by config flags
# `recompute_*`. We aggressively rematerialize, and store only the inputs to
# each decoder block.
#
# A note about mbridge: mbridge enables the HF checkpoint to automatically be
# converted into a megatron-backed implementation, as opposed to having to
# ship a separate megatron implementation for every model.

OFFLOAD=True
TP_SIZE=8
CP_SIZE=1
PP_SIZE=1
VPP_SIZE=null   # Circular repeat; not compatible with mbridge so set to null.
EP_SIZE=8
ETP_SIZE=1

ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.megatron.grad_offload=$OFFLOAD \
    actor_rollout_ref.actor.megatron.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True"

ACTOR_CONFIG="
    actor_rollout_ref.model.path=$HUGGINGFACE_MODEL_PATH \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$actor_lr_warmup_steps \
    actor_rollout_ref.actor.optim.weight_decay=$actor_weight_decay \
    actor_rollout_ref.actor.optim.clip_grad=$actor_clip_grad \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.entropy_coeff=$actor_entropy_coeff \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu"

# ===================================== Reference model =====================================
# The reference model uses the same Megatron parallelism as the actor, and is
# host-offloaded when not in use. Its log-prob forward pass uses the same
# fixed micro-batch size (in sequences per GPU) as the actor and the rollout
# log-prob path, so the per-microbatch GPU work is identical across the three
# train-side phases.

REF_CONFIG="
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.ref.megatron.param_offload=$OFFLOAD \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$infer_logprob_micro_batch_size_per_gpu"

# ===================================== Inference (vLLM) =====================================
# TP=4 for inference (gen_tp). Colocated with the actor — VERL will fill the
# remaining GPUs with a second vLLM replica.
#
# MoE layout (vLLM): enable expert parallel with EP=TP. VERL requires
# expert_parallel_size == tensor_model_parallel_size * data_parallel_size.
# Pure TP=4 would shard moe_intermediate_size=768 into 192/rank, which fails
# the FlashInfer SM100 MoE kernel alignment check (must be divisible by 128)
# on B200. EP=4 keeps full expert weights per rank instead.
#
# Note: moe_tensor_parallel_size is TRT-LLM only; do not set it for vLLM.
#
# A note about gpu_memory_utilization: This is the amount of memory/GPU
# preallocated to hold the KV cache. We set to 0.5, matching the upstream
# config.

rollout_name=vllm
infer_tp=4
# infer_ep=$infer_tp   # EP must equal TP * DP (DP defaults to 1)
# actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
gpu_memory_utilization=0.5

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 8)) \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$infer_logprob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# ===================================== Profiling =====================================
# Split into two runs — simultaneous nsys (Megatron) + torch (vLLM) hits
# CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED on the same GPUs.
#
#   nsys  — Megatron actor/ref only; rollout profiler off (default, run 1)
#           discrete=False => one .nsys-rep per GPU rank for the full step
#   torch — vLLM rollout only; actor/ref profiler off (run 2)
#   none  — profiling disabled
#
# Reports: ${EXP_DIR}/nsys_profiles/ (nsys) or ${EXP_DIR}/torch_profiles/ (torch)

NSYS_DIR="${EXP_DIR}/nsys_profiles"
TORCH_PROFILE_DIR="${EXP_DIR}/torch_profiles"
PROFILER_STEPS='[5]'
mkdir -p "${NSYS_DIR}" "${TORCH_PROFILE_DIR}"

case "${PROFILE_MODE}" in
  nsys)
    PROFILER_CONFIG="
    global_profiler.tool=nsys \
    global_profiler.save_path=$EXP_DIR \
    global_profiler.steps=${PROFILER_STEPS} \
    global_profiler.profile_continuous_steps=False \
    global_profiler.global_tool_config.nsys.discrete=False \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.ref.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.enable=False \
    actor_rollout_ref.actor.profiler.all_ranks=True \
    actor_rollout_ref.ref.profiler.all_ranks=True \
    actor_rollout_ref.actor.profiler.save_path=${NSYS_DIR} \
    actor_rollout_ref.ref.profiler.save_path=${NSYS_DIR}"
    ;;
  torch)
    PROFILER_CONFIG="
    global_profiler.tool=torch \
    global_profiler.save_path=$EXP_DIR \
    global_profiler.steps=${PROFILER_STEPS} \
    global_profiler.profile_continuous_steps=False \
    actor_rollout_ref.actor.profiler.enable=False \
    actor_rollout_ref.ref.profiler.enable=False \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.all_ranks=True \
    actor_rollout_ref.rollout.profiler.tool=torch \
    actor_rollout_ref.rollout.profiler.save_path=${TORCH_PROFILE_DIR} \
    actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True \
    actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cuda,cpu,stack]"
    ;;
  none)
    PROFILER_CONFIG="
    global_profiler.steps=null \
    actor_rollout_ref.actor.profiler.enable=False \
    actor_rollout_ref.ref.profiler.enable=False \
    actor_rollout_ref.rollout.profiler.enable=False"
    ;;
  *)
    echo "Unknown PROFILE_MODE=${PROFILE_MODE}; use nsys, torch, or none" >&2
    exit 1
    ;;
esac

# ===================================== Ray runtime env =====================================
# Env vars that need to be set on every Ray worker (not just the driver).
# Plain `export`s in this script only affect the driver process; Ray workers
# spawned by VERL will not inherit them reliably in multi-node / attached-
# cluster setups. We pass them through VERL's `ray_kwargs.ray_init.runtime_env`
# so Ray ships them to every worker it spawns. These env vars are inherited 
# from Yan Bai's reference script.
#
#   - CUDA_DEVICE_MAX_CONNECTIONS=1: required for correct CUDA stream ordering
#     with Megatron's overlapped comm/compute; must be set before CUDA init.
#   - NCCL_NVLS_ENABLE=0: disables NVLink SHARP; read by NCCL at comm init.
#   - ROCR_VISIBLE_DEVICES='': something in this environment exports the AMD/ROCm
#     device var even on these NVIDIA nodes. verl raises if both ROCR_* and
#     CUDA_VISIBLE_DEVICES are set (single_controller/base/worker.py), and Ray
#     sets CUDA_VISIBLE_DEVICES per actor. Forcing ROCR empty on every worker
#     makes verl's `if rocr_val:` guard treat it as unset.
#
# The override is passed as a single inline dict because `ray_init` is a
# declared (struct-mode) node in ppo_trainer.yaml, which blocks adding new
# nested keys via dotted paths. Values are quoted strings because Ray requires
# env_vars values to be strings.

RAY_KWARGS="+ray_kwargs.ray_init.runtime_env={env_vars:{CUDA_DEVICE_MAX_CONNECTIONS:'1',NCCL_NVLS_ENABLE:'0',HF_HOME:'${HF_HOME}',ROCR_VISIBLE_DEVICES:''}}"

# ===================================== Run =====================================
# Hydra creates outputs/<date>/<time>/ and wandb creates wandb/ relative to
# CWD. Move to the per-experiment dir so both end up there alongside the log
# file (and out of the read-only verl source tree).
cd "${EXP_DIR}"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer.yaml \
    hydra.run.dir="${EXP_DIR}/hydra/$(date +%Y-%m-%d/%H-%M-%S)" \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.train_batch_size=$train_prompt_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    $ACTOR_CONFIG \
    $ACTOR_MEGATRON_CONFIG \
    $REF_CONFIG \
    $ROLLOUT_CONFIG \
    $PROFILER_CONFIG \
    $RAY_KWARGS \
    reward_model.reward_manager=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    "${JOB_INFO[@]}" 2>&1 | tee ${OUT_FILE}
