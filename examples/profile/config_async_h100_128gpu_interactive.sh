#!/usr/bin/env bash
# config_async_h100_128g: One-step-off-policy async GRPO on 128x H100 (16 nodes),
# DISAGGREGATED generator/trainer (hybrid_engine=False):
#   - Generation:  8 nodes x 8 GPU = 64 GPU, vLLM, TP=8  -> 8 generation instances
#   - Training:    8 nodes x 8 GPU = 64 GPU, Megatron TP=4, EP=8, PP=8, CP=1 -> DP=2
# The next batch is generated on the gen pool while the current batch trains on
# the train pool (one step off-policy); weights sync gen<-train via NCCL.
#
# Batch: 256 prompts/step -> 32 prompts per generation instance (256 / 8).
# Output sequence length (OSL): up to 16K tokens.
#
# Built from examples/profile/config_b2_interactive.sh (style, profiling, env)
# and verl/experimental/one_step_off_policy/shell/dapo_7b_math_*_64_64.sh
# (disaggregated entry point + node-level resource split).
#
# Secrets are read from the environment (not stored in env.sh). Pass at launch:
#   HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_async_h100_128gpu_interactive.sh <exp> <mode>
#
# >>> ASSUMPTIONS you likely want to confirm/override (not in the spec): <<<
#   - MODEL_PATH        (default Qwen3-30B-A3B-Base, as in config_b2; PP=8 also
#                        suits a larger MoE — override for e.g. Qwen3-235B-A22B)
#   - n_resp_per_prompt (GRPO group size; default 16)
#   - max_prompt_length (default 2048)
#   - dataset gsm8k will NOT actually emit 16K-token outputs; swap in a
#     long-output dataset for a realistic 16K OSL distribution.
#   - ppo_micro_batch_size_per_gpu (default 1; 16K sequences are large — tune)

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/env.sh" ] && source "${SCRIPT_DIR}/env.sh"

# Require secrets from the environment. Validate with xtrace off so the token
# values are never echoed into the tee'd log.
{ set +x; } 2>/dev/null
: "${HF_TOKEN:?not set — pass it at invocation, e.g. HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_async_h100_128gpu_interactive.sh <exp> <mode>}"
: "${WANDB_API_KEY:?not set — pass it at invocation, e.g. HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_async_h100_128gpu_interactive.sh <exp> <mode>}"
set -x

# Driver-side env vars (also shipped to Ray workers via $RAY_KWARGS below).
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=${VERL_SRC_DIR}:${PYTHONPATH:-}

export HF_HOME="${HF_HOME:-/opt/hf-cache}"
mkdir -p $HF_HOME

# ===================================== Topology (disaggregated) =====================================
# 128 H100 total = 64 generation + 64 training, split at NODE granularity.
NGPUS_PER_NODE=8
NNODES_ROLLOUT=8   # 8 generation nodes  -> 64 GPU (vLLM, TP=8 -> 8 instances)
NNODES_TRAIN=8     # 8 training nodes    -> 64 GPU (Megatron TP=4 x PP=8 x CP=1 x DP=2)

# ===================================== Output / metadata =====================================
# ./config_async_h100_128gpu_interactive.sh <exp_base> <nsys|torch|none>  (default nsys)
WANDB_PROJECT_NAME="verl_grpo_async_h100"
EXPERIMENT_NAME_BASE="${1:-config_async_h100_128g-qwen_3_30b_a3b_base-gsm8k-$(date +%Y_%m_%d)}"
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
git config --global --add safe.directory "${VERL_SRC_DIR}"
VERL_COMMIT=$(git -C "${VERL_SRC_DIR}" rev-parse HEAD)
echo "Using VERL Commit: ${VERL_COMMIT}"

JOB_INFO=(
    "+job_info.config_number='async-h100-128g'"
    "+job_info.cluster='computelab'"
    "+job_info.gpu='H100-SXM'"
    "+job_info.num_gpus=$((NGPUS_PER_NODE * (NNODES_ROLLOUT + NNODES_TRAIN)))"
    "+job_info.gen_gpus=$((NGPUS_PER_NODE * NNODES_ROLLOUT))"
    "+job_info.train_gpus=$((NGPUS_PER_NODE * NNODES_TRAIN))"
    "+job_info.verl_container_image='${VERL_CONTAINER_IMAGE:-unknown}'"
    "+job_info.verl_commit=${VERL_COMMIT}"
    "+job_info.environment_name='gsm8k'"
    "+job_info.max_num_turns=1"
    "+job_info.environment_type='toy,math'"
    "+job_info.architecture_name='Qwen3-30B-A3B-Base'"
    "+job_info.num_params='30B'"
    "+job_info.algorithm_name='grpo'"
    "+job_info.colocated=false"
    "+job_info.async_lag=1"
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

HUGGINGFACE_MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}"
hf download "${HUGGINGFACE_MODEL_PATH}"

# ===================================== Algorithm =====================================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

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
top_k=-1
val_temperature=1.0
val_top_p=0.7
val_top_k=-1

# ===================================== Data / batch =====================================
# 256 prompts/step; with 8 generation instances that is 32 prompts/instance.
max_prompt_length=2048
max_response_length=$((1024 * 16))   # 16K OSL (max)

train_prompt_bsz=256
train_prompt_mini_bsz=32
n_resp_per_prompt=16                 # GRPO group size (NOT in spec — adjust)
n_resp_per_prompt_val=1

use_dynamic_bsz=False
ppo_micro_batch_size_per_gpu=1       # 16K seqs are large; tune up if memory allows
infer_logprob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}

# ===================================== Trainer Megatron parallelism =====================================
# 64 training GPU = TP*PP*CP*DP = 4*8*1*2. DP=2 is implicit (world/(TP*PP*CP)).
# Expert grid: EP*ETP must equal TP*CP*DP = 8 -> EP=8, ETP=1.
# Dedicated training nodes (disaggregated), so no colocation memory pressure:
# keep the actor resident on-GPU; offload only the (idle) reference model.
OFFLOAD_ACTOR=False
OFFLOAD_REF=True
TP_SIZE=4
CP_SIZE=1
PP_SIZE=8
VPP_SIZE=null   # not compatible with mbridge
EP_SIZE=8
ETP_SIZE=1

# MoE token dispatcher. On H100/Hopper the prebuilt DeepEP kernels are generally
# available (unlike B200/SM100), but default to Megatron alltoall for safety;
# set USE_DEEPEP=1 to use the DeepEP/flex path on both actor and ref.
USE_DEEPEP="${USE_DEEPEP:-0}"
if [[ "${USE_DEEPEP}" == "1" ]]; then
    ACTOR_MOE_DISPATCHER="+actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex"
    REF_MOE_DISPATCHER="+actor_rollout_ref.ref.megatron.override_transformer_config.moe_enable_deepep=True +actor_rollout_ref.ref.megatron.override_transformer_config.moe_token_dispatcher_type=flex"
else
    ACTOR_MOE_DISPATCHER="+actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall"
    REF_MOE_DISPATCHER="+actor_rollout_ref.ref.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall"
fi

ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=$OFFLOAD_ACTOR \
    actor_rollout_ref.actor.megatron.grad_offload=$OFFLOAD_ACTOR \
    actor_rollout_ref.actor.megatron.optimizer_offload=$OFFLOAD_ACTOR \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    ${ACTOR_MOE_DISPATCHER} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True"

ACTOR_CONFIG="
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.hybrid_engine=False \
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
# Same Megatron parallelism as the actor; host-offloaded when idle.
REF_CONFIG="
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.ref.megatron.param_offload=$OFFLOAD_REF \
    ${REF_MOE_DISPATCHER} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$infer_logprob_micro_batch_size_per_gpu"

# ===================================== Generation (vLLM) =====================================
# 64 dedicated generation GPU, TP=8 -> 8 vLLM instances (one per node).
# 256 prompts / 8 instances = 32 prompts per instance.
rollout_name=vllm
gen_tp=8
gpu_memory_utilization=0.8   # dedicated gen GPUs (not colocated)

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
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
# Two-run split (same rationale as config_b2 — concurrent nsys+torch on the same
# GPUs hits CUPTI_ERROR_MULTIPLE_SUBSCRIBERS):
#   nsys  — Megatron actor/ref on the TRAIN nodes (default, run 1)
#   torch — vLLM rollout on the GEN nodes (run 2)
#   none  — profiling off
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
# Shipped to every Ray worker. ROCR_VISIBLE_DEVICES='' avoids verl's clash guard
# (worker.py) when the environment exports the AMD/ROCm var on NVIDIA nodes.
RAY_KWARGS="+ray_kwargs.ray_init.runtime_env={env_vars:{CUDA_DEVICE_MAX_CONNECTIONS:'1',NCCL_NVLS_ENABLE:'0',HF_HOME:'${HF_HOME}',ROCR_VISIBLE_DEVICES:''}}"

# ===================================== Run =====================================
# cd to the per-experiment dir so wandb's relative wandb/ lands here; Hydra's run
# dir is pinned absolute below so it never depends on CWD being writable.
cd "${EXP_DIR}"

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path=config \
    --config-name=one_step_off_ppo_megatron_trainer.yaml \
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
    reward.reward_manager.name=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.nnodes=$NNODES_TRAIN \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    rollout.nnodes=$NNODES_ROLLOUT \
    rollout.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    "${JOB_INFO[@]}" 2>&1 | tee ${OUT_FILE}
